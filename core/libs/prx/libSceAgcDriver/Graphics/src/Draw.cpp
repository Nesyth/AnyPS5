#include "prx/libSceAgcDriver/Graphics/include/BindlessHeap.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MemoryAccounting.hpp"
#include "prx/libc/include/RunReport.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetTransfer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GpuColorTransfer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VertexPlanScratch.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DeviceImageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GpuZones.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuardSpans.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/StagedMerge.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/SampleOrder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libSceAgcDriver/Graphics/include/WriteValidation.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/TracyZone.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileCounter.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadScratch.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Optimization/include/Optimization/ShaderStageInputInfo.hpp"
#include "SpirvBackend/SpirvAnalysis.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <functional>
#include <optional>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <tuple>
#include <unordered_map>
#include <type_traits>
#include <utility>

namespace AgcDriver::Graphics {

void WriteCheckpointMarker(const Context& context, VkCommandBuffer commands, std::uint64_t serial, std::uint32_t queue);
void WriteNvCheckpointToken(const Context& context, VkCommandBuffer commands, std::uint64_t token, std::uint32_t queue);

namespace {
enum class VkDrawKind : std::size_t { Direct, Indexed, Indirect, Mesh, Count };
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(VkDrawKind::Count)> vkDrawCounts{};
void CountVkDraw(VkDrawKind kind) {
    vkDrawCounts[static_cast<std::size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
}


bool DrawFaultTokens() {
    static const bool enabled = std::getenv("APS5_GPU_CHECKPOINTS") != nullptr;
    return enabled;
}

void noteDrawFaultToken(const Context& context, Recorder& recorder, VkCommandBuffer commands, std::span<const CompiledShader> shaders, const Pm4::DrawParameters& draw, std::uint32_t stage, const Pm4::DrawParameters::IndirectDraw* args) {
    try {
        if (!DrawFaultTokens()) return;
        const auto queue = GuestMemory::GpuLockThreadTag();
        if (queue != 0) return;
        const auto faultToken = AgcDriver::FaultTokenRing::Instance().Claim();
        if (faultToken == 0) return;
        if (auto* entry = AgcDriver::FaultTokenRing::Instance().Mutable(faultToken)) {
            entry->submitSerial = recorder.OpenSerial();
            entry->programAddress = 0;
            std::uint64_t variant = 0;
            for (const auto& shader : shaders) {
                if (shader.program != nullptr && (variant == 0 || shader.stage == ShaderRecompiler::ShaderStage::Fragment)) variant = shader.program->variantId;
            }
            entry->variantId = variant;
            entry->stage = stage;
            entry->argBuffer = 0;
            entry->argOffset = 0;
            entry->argStride = 0;
            entry->argCount = 0;
            for (auto& word : entry->argWords) word = 0;
            if (args != nullptr) {
                entry->argBuffer = args->arguments;
                entry->argOffset = 0;
                entry->argStride = args->stride;
                try {
                    constexpr std::uint32_t argWordLimit = 4;
                    const auto first = Pm4::ReadDrawArgumentsUnsynced(*args, 0);
                    entry->argWords[0] = first.count;
                    entry->argWords[1] = first.instances;
                    entry->argWords[2] = first.firstVertexOrIndex;
                    entry->argWords[3] = draw.indexed ? first.vertexOffset : first.firstInstance;
                    entry->argCount = argWordLimit;
                } catch (...) {
                    entry->argCount = 0;
                }
            }
            std::snprintf(entry->desc, sizeof(entry->desc), "draw%s%s c=%u i=%u", draw.indexed ? " indexed" : "", args != nullptr ? " indirect" : "", draw.indexCount, draw.instanceCount);
            recorder.NoteFaultToken(faultToken);
            std::atomic_ref<AgcDriver::FaultToken>(entry->token).store(faultToken, std::memory_order_release);
        }
        WriteCheckpointMarker(context, commands, faultToken, queue);
        WriteNvCheckpointToken(context, commands, faultToken, queue);
    } catch (...) {
    }
}

struct PixelFunctionLdsPool {
    Context context{};
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetLayout empty = VK_NULL_HANDLE;
    VkDescriptorSetLayout bufferLayout = VK_NULL_HANDLE;
    std::uint32_t* header = nullptr;
    std::uint32_t subgroupSize = 0;
    std::uint32_t slots = 0;
    VkDeviceSize bytes = 0;
    VkDeviceAddress address = 0;
    const BindlessHeap* heapWritten = nullptr;
    bool announced = false;
};

std::mutex& PixelLdsMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<VkDevice, PixelFunctionLdsPool>& PixelLdsPools() {
    static std::unordered_map<VkDevice, PixelFunctionLdsPool> pools;
    return pools;
}

struct MeshWindowPool {
    Context context{};
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize bytes = 0;
    VkDeviceAddress address = 0;
    const BindlessHeap* heapWritten = nullptr;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
};

std::mutex& MeshWindowMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<VkDevice, MeshWindowPool>& MeshWindowPools() {
    static std::unordered_map<VkDevice, MeshWindowPool> pools;
    return pools;
}

void DestroyMeshWindow(MeshWindowPool& pool) {
    if (pool.descriptorPool != VK_NULL_HANDLE) pool.context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(pool.context.device, pool.descriptorPool, nullptr);
    if (pool.buffer != VK_NULL_HANDLE) pool.context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(pool.context.device, pool.buffer, nullptr);
    if (pool.memory != VK_NULL_HANDLE) MemoryAccounting::Free(pool.context, pool.memory);
    if (pool.layout != VK_NULL_HANDLE) pool.context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(pool.context.device, pool.layout, nullptr);
    pool = MeshWindowPool{};
}

void DestroyPixelFunctionLds(PixelFunctionLdsPool& pool) {
    if (pool.header != nullptr) pool.context.Function<PFN_vkUnmapMemory>("vkUnmapMemory")(pool.context.device, pool.memory);
    if (pool.descriptorPool != VK_NULL_HANDLE) pool.context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(pool.context.device, pool.descriptorPool, nullptr);
    if (pool.buffer != VK_NULL_HANDLE) pool.context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(pool.context.device, pool.buffer, nullptr);
    if (pool.memory != VK_NULL_HANDLE) MemoryAccounting::Free(pool.context, pool.memory);
    if (pool.bufferLayout != VK_NULL_HANDLE) pool.context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(pool.context.device, pool.bufferLayout, nullptr);
    if (pool.empty != VK_NULL_HANDLE) pool.context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(pool.context.device, pool.empty, nullptr);
    pool.header = nullptr;
    pool.descriptorPool = VK_NULL_HANDLE;
    pool.set = VK_NULL_HANDLE;
    pool.buffer = VK_NULL_HANDLE;
    pool.memory = VK_NULL_HANDLE;
    pool.bufferLayout = VK_NULL_HANDLE;
    pool.empty = VK_NULL_HANDLE;
}

}

PixelFunctionLdsLayouts PixelFunctionLdsSetLayouts(const Context& context) {
    std::lock_guard lock(PixelLdsMutex());
    auto& pool = PixelLdsPools()[context.device];
    if (pool.bufferLayout == VK_NULL_HANDLE) {
        pool.context = context;
        VkDescriptorSetLayoutCreateInfo emptyInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &emptyInfo, nullptr, &pool.empty), "vkCreateDescriptorSetLayout pixel function LDS empty");
        const VkDescriptorSetLayoutBinding binding{0u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = 1u;
        info.pBindings = &binding;
        Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &info, nullptr, &pool.bufferLayout), "vkCreateDescriptorSetLayout pixel function LDS");
    }
    return {pool.empty, pool.bufferLayout};
}

void PreparePixelFunctionLds(const Context& context, VkCommandBuffer commands, std::uint32_t subgroupSize) {
    const auto operations = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT;
    Require((context.subgroup.supportedStages & VK_SHADER_STAGE_FRAGMENT_BIT) != 0 && (context.subgroup.supportedOperations & operations) == operations, "pixel function LDS needs fragment subgroup elect, ballot and broadcast");
    Require(subgroupSize != 0u && subgroupSize == context.subgroup.subgroupSize, "pixel function LDS subgroup size does not match the device");
    std::lock_guard lock(PixelLdsMutex());
    auto& pool = PixelLdsPools()[context.device];
    Require(pool.bufferLayout != VK_NULL_HANDLE, "pixel function LDS descriptor set layout was not created");
    if (pool.buffer == VK_NULL_HANDLE) {
        const auto stride = static_cast<std::uint64_t>(subgroupSize) * ShaderRecompiler::FunctionLdsDwordLimit * 4ull;
        const auto budget = std::min<std::uint64_t>(context.limits.maxStorageBufferRange, 2ull << 30);
        Require(budget > 16ull + stride, "pixel function LDS does not fit in the device storage buffer range");
        pool.slots = static_cast<std::uint32_t>((budget - 16ull) / stride);
        pool.subgroupSize = subgroupSize;
        const auto bytes = 16ull + stride * pool.slots;
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = bytes;
        const bool addressed = context.descriptorHeap && context.bufferDeviceAddress;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | (addressed ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0u);
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &bufferInfo, nullptr, &pool.buffer), "vkCreateBuffer pixel function LDS");
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, pool.buffer, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        const VkMemoryAllocateFlagsInfo addressFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
        if (addressed) allocation.pNext = &addressFlags;
        allocation.allocationSize = requirements.size;
        constexpr auto hostCoherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        constexpr auto deviceLocalHost = hostCoherent | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        std::optional<std::uint32_t> deviceLocalType;
        std::optional<std::uint32_t> hostOnlyType;
        for (std::uint32_t i = 0; i < context.memory.memoryTypeCount; ++i) {
            if ((requirements.memoryTypeBits & (1u << i)) == 0) continue;
            const auto flags = context.memory.memoryTypes[i].propertyFlags;
            const auto heap = context.memory.memoryTypes[i].heapIndex;
            if (!deviceLocalType && (flags & deviceLocalHost) == deviceLocalHost && context.memory.memoryHeaps[heap].size >= requirements.size) {
                deviceLocalType = i;
            }
            if (!hostOnlyType && (flags & hostCoherent) == hostCoherent && (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) {
                hostOnlyType = i;
            }
        }
        const auto hostType = hostOnlyType.value_or(context.MemoryType(requirements.memoryTypeBits, hostCoherent));
        allocation.memoryTypeIndex = deviceLocalType.value_or(hostType);
        auto allocated = MemoryAccounting::Allocate(context, allocation, "pixel function LDS", &pool.memory);
        if (allocated != VK_SUCCESS && deviceLocalType && allocation.memoryTypeIndex == *deviceLocalType && hostType != *deviceLocalType) {
            allocation.memoryTypeIndex = hostType;
            allocated = MemoryAccounting::Allocate(context, allocation, "pixel function LDS", &pool.memory);
        }
        if (allocated != VK_SUCCESS) {
            context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, pool.buffer, nullptr);
            pool.buffer = VK_NULL_HANDLE;
            pool.memory = VK_NULL_HANDLE;
            Check(allocated, "vkAllocateMemory pixel function LDS");
        }
        Check(context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(context.device, pool.buffer, pool.memory, 0), "vkBindBufferMemory pixel function LDS");
        pool.bytes = bytes;
        if (addressed) {
            pool.address = BufferDeviceAddress(context, pool.buffer);
        }
        Check(context.Function<PFN_vkMapMemory>("vkMapMemory")(context.device, pool.memory, 0, 16, 0, reinterpret_cast<void**>(&pool.header)), "vkMapMemory pixel function LDS");
        pool.header[0] = 0u;
        pool.header[1] = 0u;
        pool.header[2] = pool.slots;
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1u;
        poolInfo.poolSizeCount = 1u;
        poolInfo.pPoolSizes = &size;
        Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool.descriptorPool), "vkCreateDescriptorPool pixel function LDS");
        VkDescriptorSetAllocateInfo sets{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        sets.descriptorPool = pool.descriptorPool;
        sets.descriptorSetCount = 1u;
        sets.pSetLayouts = &pool.bufferLayout;
        Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &sets, &pool.set), "vkAllocateDescriptorSets pixel function LDS");
        const VkDescriptorBufferInfo buffer{pool.buffer, 0, bytes};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = pool.set;
        write.dstBinding = 0u;
        write.descriptorCount = 1u;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer;
        context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 1u, &write, 0u, nullptr);
        std::fprintf(stderr, "[gpu] pixel function LDS pool: %u subgroups, %llu MiB, stride %u lanes\n", pool.slots, static_cast<unsigned long long>(bytes >> 20u), subgroupSize);
        pool.announced = true;
    }
    Require(pool.subgroupSize == subgroupSize, "pixel function LDS subgroup size does not match the pool");
    if (context.descriptorHeap && DescriptorHeapEnabled()) {
        auto& heap = BindlessHeap::For(context);
        if (pool.heapWritten != &heap) {
            Require(pool.address != 0, "the pixel function LDS pool has no device address for the GPU heap");
            heap.WritePixelLdsBuffer(pool.address, pool.bytes);
            pool.heapWritten = &heap;
        }
    }
    const VkMemoryBarrier beforeFill{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
    context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &beforeFill, 0, nullptr, 0, nullptr);
    context.Resolved(&DeviceFunctions::cmdFillBuffer, "vkCmdFillBuffer")(commands, pool.buffer, 0, 4, 0u);
    const VkMemoryBarrier afterFill{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &afterFill, 0, nullptr, 0, nullptr);
}

bool UsesMeshWindow(std::span<const CompiledShader> shaders) {
    return std::ranges::any_of(shaders, [](const CompiledShader& shader) { return shader.program != nullptr && shader.program->meshWindowBuffer; });
}

VkDescriptorSetLayout MeshWindowSetLayout(const Context& context) {
    std::lock_guard lock(MeshWindowMutex());
    auto& pool = MeshWindowPools()[context.device];
    if (pool.layout == VK_NULL_HANDLE) {
        pool.context = context;
        const VkDescriptorSetLayoutBinding binding{0u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr};
        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = 1u;
        info.pBindings = &binding;
        Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &info, nullptr, &pool.layout), "vkCreateDescriptorSetLayout mesh window");
    }
    return pool.layout;
}

void WriteMeshWindowHeap(const Context& context, MeshWindowPool& pool) {
    if (!context.descriptorHeap || !DescriptorHeapEnabled()) return;
    auto& heap = BindlessHeap::For(context);
    if (pool.heapWritten == &heap) return;
    Require(pool.address != 0, "the mesh window pool has no device address for the GPU heap");
    heap.WriteMeshWindowBuffer(pool.address, pool.bytes);
    pool.heapWritten = &heap;
}

void PrepareMeshWindow(const Context& context) {
    std::lock_guard lock(MeshWindowMutex());
    auto& pool = MeshWindowPools()[context.device];
    Require(pool.layout != VK_NULL_HANDLE, "mesh window descriptor set layout was not created");
    if (pool.set != VK_NULL_HANDLE) {
        WriteMeshWindowHeap(context, pool);
        return;
    }
    try {
        const auto bytes = (static_cast<std::uint64_t>(ShaderRecompiler::MeshWindowHeaderDwords) + static_cast<std::uint64_t>(ShaderRecompiler::MeshWindowSlots) * ShaderRecompiler::MeshWindowSlotDwords) * 4ull;
        Require(bytes <= context.limits.maxStorageBufferRange, "mesh window pool does not fit in the device storage buffer range");
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = bytes;
        const bool addressed = context.descriptorHeap && context.bufferDeviceAddress;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | (addressed ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0u);
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &bufferInfo, nullptr, &pool.buffer), "vkCreateBuffer mesh window");
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, pool.buffer, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        const VkMemoryAllocateFlagsInfo addressFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
        if (addressed) allocation.pNext = &addressFlags;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Check(MemoryAccounting::Allocate(context, allocation, "mesh window", &pool.memory), "vkAllocateMemory mesh window");
        Check(context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(context.device, pool.buffer, pool.memory, 0), "vkBindBufferMemory mesh window");
        pool.bytes = bytes;
        if (addressed) {
            pool.address = BufferDeviceAddress(context, pool.buffer);
        }
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1u;
        poolInfo.poolSizeCount = 1u;
        poolInfo.pPoolSizes = &size;
        Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool.descriptorPool), "vkCreateDescriptorPool mesh window");
        VkDescriptorSetAllocateInfo sets{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        sets.descriptorPool = pool.descriptorPool;
        sets.descriptorSetCount = 1u;
        sets.pSetLayouts = &pool.layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &sets, &set), "vkAllocateDescriptorSets mesh window");
        const VkDescriptorBufferInfo buffer{pool.buffer, 0, bytes};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set;
        write.dstBinding = 0u;
        write.descriptorCount = 1u;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer;
        context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 1u, &write, 0u, nullptr);
        CommandBatch batch(context);
        const auto commands = batch.Handle();
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, pool.buffer, 0, ShaderRecompiler::MeshWindowHeaderDwords * 4u, 0u);
        const VkMemoryBarrier afterFill{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &afterFill, 0, nullptr, 0, nullptr);
        batch.SubmitAndWait();
        pool.set = set;
        WriteMeshWindowHeap(context, pool);
        std::fprintf(stderr, "[gpu] mesh window pool: %u workgroup slots, %llu MiB\n", ShaderRecompiler::MeshWindowSlots, static_cast<unsigned long long>(bytes >> 20u));
    } catch (...) {
        const auto layout = pool.layout;
        pool.layout = VK_NULL_HANDLE;
        DestroyMeshWindow(pool);
        pool.context = context;
        pool.layout = layout;
        throw;
    }
}

void BindMeshWindow(const Context& context, VkCommandBuffer commands, VkPipelineLayout layout) {
    std::lock_guard lock(MeshWindowMutex());
    const auto& pool = MeshWindowPools().at(context.device);
    Require(pool.set != VK_NULL_HANDLE, "mesh window buffer was not prepared");
    CmdBindDescriptorSets(context, commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, ShaderRecompiler::MeshWindowDescriptorSet, 1, &pool.set, SetBindSite::Draw);
}

void ReleaseMeshWindow(VkDevice device) {
    std::lock_guard lock(MeshWindowMutex());
    const auto found = MeshWindowPools().find(device);
    if (found == MeshWindowPools().end()) return;
    DestroyMeshWindow(found->second);
    MeshWindowPools().erase(found);
}

std::uint32_t PixelFunctionLdsSubgroup(std::span<const CompiledShader> shaders) {
    std::uint32_t subgroup = 0u;
    for (const auto& shader : shaders) {
        if (shader.program != nullptr && shader.program->functionLdsBuffer) subgroup = shader.program->hostSubgroupSize;
    }
    return subgroup;
}

void BindPixelFunctionLds(const Context& context, VkCommandBuffer commands, VkPipelineLayout layout) {
    std::lock_guard lock(PixelLdsMutex());
    const auto& pool = PixelLdsPools().at(context.device);
    Require(pool.set != VK_NULL_HANDLE, "pixel function LDS buffer was not prepared");
    CmdBindDescriptorSets(context, commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, ShaderRecompiler::PixelFunctionLdsDescriptorSet, 1, &pool.set, SetBindSite::Draw);
}

void CheckPixelFunctionLds(const Context& context) {
    std::lock_guard lock(PixelLdsMutex());
    const auto found = PixelLdsPools().find(context.device);
    if (found == PixelLdsPools().end() || found->second.header == nullptr) return;
    const auto exhausted = found->second.header[1] != 0u;
    found->second.header[1] = 0u;
    if (exhausted) throw std::runtime_error("AGC graphics: pixel function LDS subgroup slots exhausted");
}

void ReleasePixelFunctionLds(VkDevice device) {
    std::lock_guard lock(PixelLdsMutex());
    const auto found = PixelLdsPools().find(device);
    if (found == PixelLdsPools().end()) return;
    DestroyPixelFunctionLds(found->second);
    PixelLdsPools().erase(found);
}

std::uint64_t DrawRenderPassKey(const Context& context, const State& state, std::span<const VkImageView> targetViews) {
    std::uint64_t key = 14695981039346656037ull;
    const auto mix = [&](std::uint64_t value) {
        key ^= value;
        key *= 1099511628211ull;
    };
    for (const auto view : targetViews) mix(reinterpret_cast<std::uint64_t>(view));
    if (state.blends.size() != state.colors.size()) {
        mix(state.blends.size());
        for (const auto& color : state.colors) mix(color.exportIndex);
    }
    for (const auto& color : state.colors) {
        if (color.partialWriteMask != 0u) mix(0x6d65726765000000ull | color.exportIndex);
    }
    mix(state.renderExtent.width);
    mix(state.renderExtent.height);
    mix(state.samples);
    mix(state.mergedCoverage);
    mix(state.truncatedCoverage);
    mix(state.customSampleLocations);
    if (state.customSampleLocations) {
        mix(state.sampleLocationGrid);
        for (const auto location : state.sampleLocations) mix(location);
    }
    if (!context.provokingVertexModePerPipeline) mix(state.provokingVertexMode);
    if (state.rasterizerDiscard) mix(0x6469736361726400ull);
    return key;
}

namespace {

bool writesStencilTarget(const State& state) {
    const auto keeps = [](const VkStencilOpState& face) {
        return face.writeMask == 0u || (face.failOp == VK_STENCIL_OP_KEEP && face.passOp == VK_STENCIL_OP_KEEP && face.depthFailOp == VK_STENCIL_OP_KEEP);
    };
    return state.stencilTest && (!keeps(state.stencilFront) || !keeps(state.stencilBack));
}

bool writesDepthTarget(const State& state) {
    return state.depthWrite || writesStencilTarget(state);
}

std::optional<std::uint8_t> overwrittenStencil(const State& state) {
    const auto replaces = [](const VkStencilOpState& face) { return face.compareOp == VK_COMPARE_OP_ALWAYS && face.passOp == VK_STENCIL_OP_REPLACE && face.writeMask == 0xffu; };
    if (!state.depthTest || !state.depthWrite || state.depthCompare != VK_COMPARE_OP_ALWAYS || state.depthBoundsTest || !state.stencilTest) return std::nullopt;
    if (!replaces(state.stencilFront) || !replaces(state.stencilBack) || state.stencilFront.reference != state.stencilBack.reference) return std::nullopt;
    return static_cast<std::uint8_t>(state.stencilFront.reference);
}

std::size_t colorKeyCount(const ColorTarget& color, std::size_t bytes) {
    if (color.mipCount != 1 || color.depth != 1 || !color.dccPipeAligned) return DccKeyBytes(bytes);
    return DccKeyCount(ColorTextureTileMode(color.tileMode), color.elementBytes, color.extent.width, color.extent.height, bytes);
}

std::array<std::byte, 16> clearTexel(const ColorTarget& color, DccKeys keys) {
    std::array<std::byte, 16> texel{};
    const auto elementBytes = static_cast<std::size_t>(color.elementBytes);
    Require(elementBytes != 0 && elementBytes <= texel.size(), "unexpected color element size");
    if (keys == DccKeys::ClearRegister) {
        Require(elementBytes <= sizeof(color.clearWords), "the DCC register clear of a texel over 64 bits is not modeled");
        std::memcpy(texel.data(), color.clearWords.data(), elementBytes);
    } else {
        Require(FillDccClear(color.format, keys, color.dccAlphaOnMsb, std::span(texel.data(), elementBytes)), "the target's format has no encoding of its DCC clear code");
    }
    return texel;
}

bool clearToTexel(StorageTexture& image, const std::array<std::byte, 16>& texel, std::uint32_t elementBytes, const char*& refusal, bool allowRegisterClear = false) {
    std::array<std::byte, 16> repeated{};
    for (std::size_t offset = 0; offset + elementBytes <= repeated.size(); offset += elementBytes) std::memcpy(repeated.data() + offset, texel.data(), elementBytes);
    std::array<std::uint32_t, 4> pattern{};
    std::memcpy(pattern.data(), repeated.data(), repeated.size());
    return image.FillClear(std::span<const std::uint32_t, 4>(pattern), StorageTexture::WholeImage, refusal, allowRegisterClear);
}

void writeTexels(const ColorTarget& color, const std::array<std::byte, 16>& texel) {
    const auto elementBytes = static_cast<std::size_t>(color.elementBytes);
    std::vector<std::byte> texels(color.bytes);
    for (std::size_t offset = 0; offset + elementBytes <= texels.size(); offset += elementBytes) std::memcpy(texels.data() + offset, texel.data(), elementBytes);
    GuestMemory::Write(color.address, texels);
}

void storeClearTexels(const Context& context, const ColorTarget& color, const std::array<std::byte, 16>& texel) {
    StorageTexture::FlushPending(color.address, color.bytes, nullptr, "fast-clear materialization");
    const auto keys = ReadDccKeys(color.dccAddress, color.bytes);
    if (!IsDccClear(keys)) return;
    writeTexels(color, keys == DccKeys::ClearRegister ? texel : clearTexel(color, keys));
    MarkDccUncompressed(context, color.dccAddress, color.bytes, colorKeyCount(color, color.bytes));
}

void storeSingleTexels(const Context& context, const ColorTarget& color) {
    Require(!ColorTileModeIsLinear(color.tileMode), "comp-to-single DCC keys of a linear color target");
    const ColorTargetLayout surface(color.extent.width, color.extent.height, color.tileMode, color.elementBytes, color.pipeBankXor);
    Require(surface.Bytes() == color.bytes, "comp-to-single DCC keys over a color target whose layout differs from its surface");
    StorageTexture::FlushPending(color.address, color.bytes, nullptr, "comp-to-single expansion");
    std::vector<std::byte> texels(color.bytes);
    GuestMemory::Read(color.address, texels, surface.Alignment());
    constexpr std::size_t BlockBytes = 256;
    for (std::uint32_t y = 0; y < color.extent.height; ++y) {
        for (std::uint32_t x = 0; x < color.extent.width; ++x) {
            const auto offset = surface.Offset(x, y);
            const auto first = offset / BlockBytes * BlockBytes;
            if (offset != first) std::memcpy(texels.data() + offset, texels.data() + first, color.elementBytes);
        }
    }
    GuestMemory::Write(color.address, texels, surface.Alignment());
    MarkDccUncompressed(context, color.dccAddress, color.bytes, colorKeyCount(color, color.bytes));
}

void materializeRegisterClear(const Context& context, const ColorTarget& color, StorageTexture& resident) {
    if (color.dccAddress == 0 || resident.Descriptor().dccAddress != color.dccAddress) return;
    if (ProvedCurrentDccKeys(color.dccAddress, color.bytes, resident.TargetKeyProof()) != DccKeys::ClearRegister) return;
    const auto texel = clearTexel(color, DccKeys::ClearRegister);
    const char* refusal = nullptr;
    const bool pendingKeysKnown = KnownPendingDccKeys(color.dccAddress, color.bytes) == DccKeys::ClearRegister;
    bool cleared = clearToTexel(resident, texel, color.elementBytes, refusal, pendingKeysKnown);
    const auto keyBytes = static_cast<std::size_t>(color.bytes / 256);
    if (!cleared && keyBytes != 0) {
        if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->PendingWriteOverlaps(color.dccAddress, keyBytes)) {
            Recorder::CountSync(2);
            recorder->SyncThrough(color.dccAddress, keyBytes);
            if (CurrentDccKeys(color.dccAddress, color.bytes) != DccKeys::ClearRegister) return;
            cleared = clearToTexel(resident, texel, color.elementBytes, refusal);
        }
    }
    if (cleared) {
        MarkDccUncompressed(context, color.dccAddress, color.bytes, colorKeyCount(color, color.bytes));
        return;
    }
    storeClearTexels(context, color, texel);
    resident.Refresh();
}

enum class CmaskState { Expanded, Cleared, Partial };

struct CmaskProof {
    std::size_t bytes = 0;
    std::uint64_t generation = 0;
};

AgcDriver::Mutex& cmaskProofMutex() {
    static AgcDriver::Mutex mutex;
    return mutex;
}

std::unordered_map<std::uint64_t, CmaskProof>& cmaskProofs() {
    static std::unordered_map<std::uint64_t, CmaskProof> proofs;
    return proofs;
}

bool cmaskProved(std::uint64_t address, std::size_t bytes) {
    if (GuestMemory::CollectWrites(address, bytes) == 0) return false;
    std::lock_guard lock(cmaskProofMutex());
    const auto found = cmaskProofs().find(address);
    return found != cmaskProofs().end() && found->second.bytes == bytes && !GuestMemory::StoredOver(address, bytes, found->second.generation);
}

void proveCmask(std::uint64_t address, std::size_t bytes, std::uint64_t generation) {
    std::lock_guard lock(cmaskProofMutex());
    if (generation == 0) cmaskProofs().erase(address);
    else cmaskProofs()[address] = {bytes, generation};
}

void requireCmaskCodes(std::uint32_t codes) {
    Require((codes & 1u) == 0, "a CMASK fast clear of a DCC color target, whose clear belongs in its DCC keys, was recorded");
}

CmaskState readCmask(const ColorTarget& color, const CmaskLayout& layout, std::vector<std::uint32_t>& cleared) {
    const auto address = color.cmaskAddress;
    const auto bytes = layout.Bytes();
    const bool locked = GuestMemory::GpuMutex().HeldByThisThread();
    auto* recorder = Recorder::Active();
    if (locked && recorder != nullptr && recorder->PendingWriteOverlaps(address, bytes)) {
        Recorder::CountSync(2);
        recorder->SyncThrough(address, bytes);
    }
    if (AnyShadowedOverlaps(address, bytes)) {
        Require(locked, "a CMASK under a unit shadow must be read under the device lock");
        PublishShadow(address, bytes, PublishScope::Whole, PublishReason::Keys);
        if (recorder != nullptr) {
            Recorder::CountSync(2);
            recorder->Sync();
        }
    }
    std::vector<std::byte> mask(bytes);
    GuestMemory::Read(address, mask, CmaskLayout::Alignment);
    const auto uniform = [&](std::byte value) { return std::all_of(mask.begin(), mask.end(), [&](std::byte entry) { return entry == value; }); };
    if (color.coverageSamples > color.samples) {
        for (std::uint32_t tileY = 0; tileY < layout.TilesY(); ++tileY) {
            for (std::uint32_t tileX = 0; tileX < layout.TilesX(); ++tileX) {
                const auto nibble = layout.Nibble(tileX, tileY);
                const auto code = (std::to_integer<std::uint32_t>(mask[nibble / 2u]) >> ((nibble % 2u) * 4u)) & 0xfu;
                if (code != 0u && (code >> 2u) != 3u) {
                    char text[128];
                    std::snprintf(text, sizeof(text), "AGC graphics: CMASK code 0x%x of a one-fragment color target is not implemented", code);
                    throw std::runtime_error(text);
                }
            }
        }
    }
    if (uniform(std::byte{0xff})) return CmaskState::Expanded;
    if (uniform(std::byte{0})) return CmaskState::Cleared;
    for (std::uint32_t tileY = 0; tileY < layout.TilesY(); ++tileY) {
        for (std::uint32_t tileX = 0; tileX < layout.TilesX(); ++tileX) {
            const auto nibble = layout.Nibble(tileX, tileY);
            const auto code = (std::to_integer<std::uint32_t>(mask[nibble / 2u]) >> ((nibble % 2u) * 4u)) & 0xfu;
            if (code == 0u) cleared.push_back(tileY * layout.TilesX() + tileX);
        }
    }
    if (cleared.empty()) return CmaskState::Expanded;
    if (cleared.size() == static_cast<std::size_t>(layout.TilesX()) * layout.TilesY()) return CmaskState::Cleared;
    return CmaskState::Partial;
}

void storeCmaskClearTexels(const ColorTarget& color, const CmaskLayout& layout, const std::vector<std::uint32_t>* tiles, const std::array<std::byte, 16>& texel) {
    StorageTexture::FlushPending(color.address, color.bytes, nullptr, "fast-clear materialization");
    const ColorTargetLayout surface(color.extent.width, color.extent.height, color.tileMode, color.elementBytes);
    Require(surface.Bytes() == color.bytes, "CMASK fast clear over a color target whose layout differs from its surface");
    std::vector<std::byte> texels(surface.Bytes());
    GuestMemory::Read(color.address, texels, surface.Alignment());
    const auto fill = [&](std::uint32_t tile) {
        const auto x0 = (tile % layout.TilesX()) * 8u;
        const auto y0 = (tile / layout.TilesX()) * 8u;
        for (auto y = y0; y < std::min(y0 + 8u, color.extent.height); ++y) {
            for (auto x = x0; x < std::min(x0 + 8u, color.extent.width); ++x) std::memcpy(texels.data() + surface.Offset(x, y), texel.data(), color.elementBytes);
        }
    };
    if (tiles != nullptr) {
        for (const auto tile : *tiles) fill(tile);
    } else {
        for (std::uint32_t tile = 0; tile < layout.TilesX() * layout.TilesY(); ++tile) fill(tile);
    }
    GuestMemory::Write(color.address, texels, surface.Alignment());
}

void recordCmaskClear(const Context& context, Recorder& recorder, const HostImport& import, const ColorTarget& color, const CmaskLayout& layout, const std::shared_ptr<StorageTexture>& resident, VkImageView view) {
    const auto address = color.cmaskAddress;
    const auto bytes = layout.Bytes();
    StorageTexture::FlushPending(address, bytes, nullptr, "fast-clear materialization");
    if (AnyShadowedOverlaps(address, bytes)) PublishShadow(address, bytes, PublishScope::Whole, PublishReason::Keys);
    recorder.FlushKeyStoresOverlapping(address, bytes);
    const auto commands = recorder.Commands();
    if (resident != nullptr && !resident->Cached()) recorder.KeepBytes(resident.get(), static_cast<std::size_t>(resident->BudgetBytes()));
    recorder.Keep(resident);
    context.detiler->DispatchCmaskClear(commands, import.buffer, HostImportOffset(import, address), bytes, view, color.extent.width, color.extent.height, color.elementBytes, color.clearWords, color.dccAddress == 0, resident.get());
    const Recorder::WriterScope writerScope{Recorder::WriterTag(2)};
    recorder.NotePendingWrite(address, bytes);
    proveCmask(address, bytes, GuestMemory::MarkWritten(address, bytes));
}

bool materializeCmaskClear(const Context& context, const ColorTarget& color, const std::shared_ptr<StorageTexture>& resident) {
    if (color.cmaskAddress == 0) return false;
    if (context.detiler != nullptr) requireCmaskCodes(context.detiler->CmaskErrors());
    const CmaskLayout layout(color.extent.width, color.extent.height);
    const auto bytes = layout.Bytes();
    Require(color.cmaskBytes == bytes, "CMASK size differs from the color target's CMASK layout");
    if (cmaskProved(color.cmaskAddress, bytes)) return false;
    if (color.coverageSamples > color.samples) {
        std::vector<std::uint32_t> codes;
        static_cast<void>(readCmask(color, layout, codes));
    }
    auto* recorder = Recorder::Active();
    if (resident != nullptr && context.detiler != nullptr && recorder != nullptr && GuestMemory::GpuMutex().HeldByThisThread()) {
        const auto* import = HostImportFor(context, color.cmaskAddress, bytes);
        const auto view = import != nullptr ? resident->ElementView() : VK_NULL_HANDLE;
        if (view != VK_NULL_HANDLE) {
            recordCmaskClear(context, *recorder, *import, color, layout, resident, view);
            return true;
        }
    }
    const auto collected = GuestMemory::CollectWrites(color.cmaskAddress, bytes);
    std::vector<std::uint32_t> cleared;
    const auto state = readCmask(color, layout, cleared);
    if (state == CmaskState::Expanded) {
        proveCmask(color.cmaskAddress, bytes, collected);
        return false;
    }
    Require(color.dccAddress == 0, "CMASK fast clear of a DCC color target, whose clear belongs in its DCC keys");
    const auto texel = clearTexel(color, DccKeys::ClearRegister);
    const char* refusal = nullptr;
    const bool filled = state == CmaskState::Cleared && color.mipCount == 1 && resident != nullptr && clearToTexel(*resident, texel, color.elementBytes, refusal);
    if (!filled) {
        storeCmaskClearTexels(color, layout, state == CmaskState::Cleared ? nullptr : &cleared, texel);
        if (resident != nullptr) resident->Refresh();
    }
    const std::vector<std::byte> expanded(bytes, std::byte{0xff});
    GuestMemory::Write(color.cmaskAddress, expanded, CmaskLayout::Alignment);
    proveCmask(color.cmaskAddress, bytes, GuestMemory::CollectWrites(color.cmaskAddress, bytes));
    return filled;
}

void imageBarrier(const Context& context, VkCommandBuffer commands, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage, VkAccessFlags sourceAccess, VkAccessFlags destinationAccess, VkImageSubresourceRange subresources = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = sourceAccess;
    barrier.dstAccessMask = destinationAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = subresources;
    context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void memoryBarrier(const Context& context, VkCommandBuffer commands, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage, VkAccessFlags sourceAccess, VkAccessFlags destinationAccess) {
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, sourceAccess, destinationAccess};
    context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, sourceStage, destinationStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// The color surface as one untailed SW_64KB_R_X mip for the GPU detiler, with tightly packed linear rows.
TileMipLayout ColorTargetMip(const ColorTarget& color, const ColorTargetLayout& layout) {
    TileMipLayout mip{};
    mip.width = color.extent.width;
    mip.height = color.extent.height;
    mip.blocksPerRow = layout.BlocksPerRow();
    mip.pitchBytes = color.extent.width * color.elementBytes;
    mip.tiledSize = layout.Bytes();
    mip.linearSize = layout.LinearBytes();
    return mip;
}

// APS5_PROFILE_DRAW: per-draw phase timers in microseconds (a recorded draw's phases are far below
// the millisecond the old print rounded to), totalled over 10 s in the [draws] line.
enum DrawPhase : std::size_t { PhaseValidate, PhaseVertex, PhaseSetup, PhaseReadTarget, PhasePrepare, PhaseLookup, PhaseResources, PhasePipeline, PhaseRecord, PhaseKeep, PhaseSync, PhaseWriteBack, PhaseDescribe, PhaseCount };
constexpr std::array<const char*, PhaseCount> DrawPhaseNames{"validate", "vertex", "setup", "readTarget", "prepare", "lookup", "resources", "pipeline", "record", "keep", "sync", "writeBack", "describe"};

// Why a draw did not go into the recorder without a wait (counted in the [draws] line): its targets
// are not all resident, no recorder is active, a switch (APS5_SYNC_DRAWS, APS5_DUMP_TARGETS,
// APS5_SYNC_COMPLETION_DRAWS) forced it, or its completion work writes copied guest buffers or
// releases an address-based build's lease, which the CPU must not outrun (see `recorded` in Draw).
enum SyncReason : std::size_t { SyncNone, SyncNotResident, SyncNoRecorder, SyncDisabled, SyncCopiedWrites, SyncLease, SyncCount };
constexpr std::array<const char*, SyncCount> SyncReasonNames{"none", "non-resident target", "no recorder", "disabled", "copied writes", "lease"};
constexpr std::size_t IndirectPathCount = static_cast<std::size_t>(IndirectDrawPath::Count);
constexpr std::array<const char*, IndirectPathCount> IndirectDrawPathNames{"gpu-side", "patched SGPR not folded", "fetch offset unknown", "non-vertex path", "GE_INDX_OFFSET", "draw index", "vertex range too large", "feature gap", "pending image results", "pending label or copied write", "not imported", "disabled", "index range not fully readable", "index limit unknown", "records settled"};

// What became of one draw, for the totals.
struct DrawOutcome {
    // In the recorder; `waited` when the recorder was synced right after (copied writes or a lease).
    bool recorded = false;
    bool waited = false;
    bool completion = false;
    // A recorded draw began a render pass of its own, or continued the previous draw's.
    bool passBegun = false;
    bool passContinued = false;
    PassBreak passBreak = PassBreak::Continued;
    bool keyStoresFlushed = false;
    bool storesFlushed = false;
    bool addressDraw = false;
    SyncReason reason = SyncNone;
    // Shader validation memo (see CachedFragmentOutputs).
    bool validateMemoized = false;
    bool validateHit = false;
    // A build that acquired the allocation registry lease (BDA), whose cost sits outside the
    // build's own sub-phases.
    bool addressBased = false;
    // Resident target lookups, and those that took a millisecond or more. Whether a slow one
    // re-uploaded the image or just walked its pages under contention is not visible from here
    // (StorageTexture::Version advances on every MarkDirty as well as on an upload, and Generation
    // is stamped afresh by every write-watch walk): the [texture] line's "reused" and "direct
    // uploads" counters and APS5_TRACE_UPLOAD name the actual uploads.
    std::uint64_t targetLookups = 0;
    std::uint64_t slowLookups = 0;
    double slowLookupUs = 0;
    // The lookups by whether the draw's scissor covers the whole target (an area-scoped proof
    // would change nothing for those), the pages their write-watch collects walked and their time.
    std::uint64_t fullScissorLookups = 0;
    std::uint64_t partialLookups = 0;
    std::uint64_t pagesWalked = 0;
    double lookupUs = 0;
    // How the draw's resources came about (DrawKind), for the per-kind averages.
    std::size_t kind = 0;
    // The flush hook's fence waits made inside this draw (index and vertex reads, the target
    // lookups' flushes, the build's texture lookups): nested in the draw's GpuMutex hold, so they
    // are the part of the hold that is waiting rather than working. The draw's own syncs (a
    // recorded-then-waited draw's recorder Sync, a synchronous draw's SubmitAndWait) are kept apart
    // in ownSyncUs: those wait by design and would otherwise count as nested hook waits.
    double hookWaitUs = 0;
    double ownSyncUs = 0;
};

// The resources of a draw: a recipe hit (reserved for the draw recipe step), a resource-cache
// template hit, a build, or an address-based (BDA) build.
enum DrawKind : std::size_t { KindRecipeHit, KindTemplateHit, KindBuild, KindBda, KindCount };
constexpr std::array<const char*, KindCount> DrawKindNames{"recipe hit", "template hit", "build", "BDA"};

struct DrawProfile {
    AgcDriver::Mutex mutex;
    std::array<double, KindCount> kindUs{};
    std::array<std::uint64_t, KindCount> kindCounts{};
    std::uint64_t fullScissorLookups = 0;
    std::uint64_t partialLookups = 0;
    std::uint64_t pagesWalked = 0;
    double lookupUs = 0;
    std::array<double, PhaseCount> totalsUs{};
    // The longest single draw's time per phase, and the longest draw: the [lock] line's 'draw'
    // hold max (tens of ms against an average well under a millisecond) needs a phase name.
    std::array<double, PhaseCount> maxUs{};
    double maxDrawUs = 0;
    double hookWaitUs = 0;
    double maxHookWaitUs = 0;
    double ownSyncUs = 0;
    // The resources build's sub-phases (ShaderResources::Timing) of draws only; dispatches report
    // theirs in the [resources] line. `other` is the rest of the build (prepareAddressBindings:
    // the registry lease and snapshots of an address-based build, the BDA table, the layout
    // lookup), and how much of it address-based builds account for.
    double bindingsUs = 0;
    double uploadUs = 0;
    double descriptorsUs = 0;
    double otherUs = 0;
    double addressOtherUs = 0;
    std::uint64_t addressBuilds = 0;
    std::uint64_t draws = 0;
    std::uint64_t recorded = 0;
    std::uint64_t waited = 0;
    // Recorded draws whose write-back (BDA fault check) runs as a completion action instead of
    // making the draw synchronous (see APS5_SYNC_COMPLETION_DRAWS).
    std::uint64_t completion = 0;
    // Render passes recorded draws began, and draws that continued the previous draw's pass.
    std::uint64_t passesBegun = 0;
    std::uint64_t passesContinued = 0;
    std::array<DrawPassCounts, 2> passTotals{};
    std::array<std::uint64_t, PassBreakCount> passBreaks{};
    std::array<std::uint64_t, PassBreakCount> reportedPassBreaks{};
    std::array<std::uint64_t, SyncCount> reasons{};
    // The CPU inside Graphics::Draw, split by outcome: a synchronous draw costs ten times a recorded
    // one, so one average would only show the synchronous population.
    double recordedUs = 0;
    double waitedUs = 0;
    double synchronousUs = 0;
    std::uint64_t targetLookups = 0;
    std::uint64_t slowLookups = 0;
    double slowLookupUs = 0;
    // Resource cache outcomes: hits, misses (built and inserted when reusable), entries that failed
    // Revalidate, and draws that could not use the cache (synchronous, no variant id, or disabled).
    // Which key words the misses differ in is the cache's own "[rescache] miss churn" line
    // (ResourceCache::noteMiss compares each miss with the last key of the same variants).
    std::uint64_t cacheHits = 0;
    std::uint64_t cacheMisses = 0;
    std::uint64_t cacheInvalidated = 0;
    std::uint64_t uncacheable = 0;
    // Shader validation memo (see CachedFragmentOutputs).
    std::uint64_t validateHits = 0;
    std::uint64_t validateMisses = 0;
    // Draw recipe outcomes (DrawWithRecipe): hits, and misses by DrawRecipeMiss.
    std::uint64_t recipeHits = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrawRecipeMiss::Count)> recipeMisses{};
    // Indirect draws by path (IndirectDrawPath), the GPU-side ones whose records were rewritten
    // with a constant, and the time of the CPU record reads (their syncs).
    std::array<std::uint64_t, IndirectPathCount> indirect{};
    std::uint64_t indirectRewritten = 0;
    std::uint64_t indirectEmpty = 0;
    double indirectReadUs = 0;
    // Draw packets that drew nothing, by DrawSkip, and their time.
    std::array<std::uint64_t, static_cast<std::size_t>(DrawSkip::Count)> skips{};
    std::array<double, static_cast<std::size_t>(DrawSkip::Count)> skipUs{};
    AddressDrawReuseCounts addressSeen{};
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};
constexpr std::array<const char*, static_cast<std::size_t>(DrawSkip::Count)> DrawSkipNames{"nothing to draw", "prechecked", "thrown"};
constexpr std::array<const char*, static_cast<std::size_t>(AddressDrawReuse::Count)> AddressDrawReuseNames{"reused", "built", "not read-only", "not space-only", "snapshot outside the space (copied)", "partial overlap", "snapshot changed", "moved GuestBuffers element", "patched data word", "space or epoch change", "in use", "revalidation"};

std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(AddressDrawReuse::Count)>& addressDrawReuses() {
    static std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(AddressDrawReuse::Count)> counts{};
    return counts;
}

void countAddressDrawReuse(AddressDrawReuse outcome) {
    addressDrawReuses()[static_cast<std::size_t>(outcome)].fetch_add(1, std::memory_order_relaxed);
}

std::atomic<std::uint64_t>& addressDrawReuseVerifications() {
    static std::atomic<std::uint64_t> verifications{0};
    return verifications;
}

void completeSnapshotWrites(std::span<const GuestMemorySnapshot> snapshots) {
    auto* recorder = Recorder::Active();
    if (recorder == nullptr || Recorder::InCompletion()) return;
    for (const auto& snapshot : snapshots) {
        if (snapshot.bytes.empty() || !recorder->PendingWriteOverlaps(snapshot.address, snapshot.bytes.size())) continue;
        Recorder::CountSync(4);
        recorder->SyncThrough(snapshot.address, snapshot.bytes.size());
    }
}

DrawProfile& Profile() {
    static DrawProfile profile;
    return profile;
}

// Adds one draw's phases to the totals and prints the [draws] and [rescache] lines every 10 s.
void reportDraw(const std::array<double, PhaseCount>& us, const ShaderResources::BuildTiming* built, const DrawOutcome& outcome) {
    auto& profile = Profile();
    std::lock_guard lock(profile.mutex);
    double drawUs = 0;
    for (std::size_t i = 0; i < PhaseCount; ++i) {
        profile.totalsUs[i] += us[i];
        profile.maxUs[i] = std::max(profile.maxUs[i], us[i]);
        drawUs += us[i];
    }
    profile.maxDrawUs = std::max(profile.maxDrawUs, drawUs);
    profile.hookWaitUs += outcome.hookWaitUs;
    profile.maxHookWaitUs = std::max(profile.maxHookWaitUs, outcome.hookWaitUs);
    profile.ownSyncUs += outcome.ownSyncUs;
    if (built != nullptr) {
        profile.bindingsUs += built->bindingsMs * 1000.0;
        profile.uploadUs += built->uploadMs * 1000.0;
        profile.descriptorsUs += built->descriptorsMs * 1000.0;
        const auto other = std::max(0.0, us[PhaseResources] - (built->bindingsMs + built->uploadMs + built->descriptorsMs) * 1000.0);
        profile.otherUs += other;
        if (outcome.addressBased) {
            profile.addressOtherUs += other;
            ++profile.addressBuilds;
        }
    }
    ++profile.draws;
    if (outcome.recorded) ++(outcome.waited ? profile.waited : profile.recorded);
    if (outcome.completion) ++profile.completion;
    if (outcome.passBegun) ++profile.passesBegun;
    if (outcome.passBegun) ++profile.passBreaks[static_cast<std::size_t>(outcome.passBreak)];
    if (outcome.passContinued) ++profile.passesContinued;
    if (outcome.recorded) {
        auto& totals = profile.passTotals[outcome.addressDraw ? 1 : 0];
        ++totals.draws;
        if (outcome.passBegun) ++totals.passesBegun;
        if (outcome.passContinued) ++totals.passesContinued;
        if (outcome.keyStoresFlushed) ++totals.keyStoreFlushes;
        if (outcome.storesFlushed) ++totals.storeFlushes;
    }
    ++profile.reasons[outcome.reason];
    (outcome.recorded ? (outcome.waited ? profile.waitedUs : profile.recordedUs) : profile.synchronousUs) += drawUs;
    profile.targetLookups += outcome.targetLookups;
    profile.slowLookups += outcome.slowLookups;
    profile.slowLookupUs += outcome.slowLookupUs;
    profile.fullScissorLookups += outcome.fullScissorLookups;
    profile.partialLookups += outcome.partialLookups;
    profile.pagesWalked += outcome.pagesWalked;
    profile.lookupUs += outcome.lookupUs;
    if (outcome.kind < KindCount) {
        ++profile.kindCounts[outcome.kind];
        profile.kindUs[outcome.kind] += drawUs;
    }
    if (outcome.validateMemoized) ++(outcome.validateHit ? profile.validateHits : profile.validateMisses);
    const auto now = std::chrono::steady_clock::now();
    if (now - profile.lastReport < std::chrono::seconds(10)) return;
    profile.lastReport = now;
    const auto synchronous = profile.draws - profile.recorded - profile.waited;
    const auto average = [](double total, std::uint64_t count) { return count != 0 ? total / static_cast<double>(count) : 0.0; };
    char line[2048];
    int n = std::snprintf(line, sizeof(line), "[draws] %llu draws over 10 s (%llu recorded avg %.0f us, of them %llu with completion; %llu recorded then waited avg %.0f us; %llu synchronous avg %.0f us; render passes %llu begun", static_cast<unsigned long long>(profile.draws), static_cast<unsigned long long>(profile.recorded), average(profile.recordedUs, profile.recorded), static_cast<unsigned long long>(profile.completion), static_cast<unsigned long long>(profile.waited), average(profile.waitedUs, profile.waited), static_cast<unsigned long long>(synchronous), average(profile.synchronousUs, synchronous), static_cast<unsigned long long>(profile.passesBegun));
    const auto room = [&] { return n > 0 && static_cast<std::size_t>(n) < sizeof(line); };
    bool listed = false;
    for (std::size_t i = 1; i < PassBreakCount && room(); ++i) {
        const auto count = profile.passBreaks[i] - profile.reportedPassBreaks[i];
        if (count == 0) continue;
        n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "%s%s %llu", listed ? ", " : " (", PassBreakNames[i], static_cast<unsigned long long>(count));
        listed = true;
    }
    if (room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "%s, %llu draws continued one; waited or synchronous because:", listed ? ")" : "", static_cast<unsigned long long>(profile.passesContinued));
    for (std::size_t i = SyncNone + 1; i < SyncCount && room(); ++i) {
        if (profile.reasons[i] != 0) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s %llu", SyncReasonNames[i], static_cast<unsigned long long>(profile.reasons[i]));
    }
    if (room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "):");
    for (std::size_t i = 0; i < PhaseCount && room(); ++i) {
        if (profile.totalsUs[i] <= 0) continue;
        n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s=%.1fms", DrawPhaseNames[i], profile.totalsUs[i] / 1000.0);
        if (i == PhaseResources && room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " (bindings %.1f, upload %.1f, descriptors %.1f, other %.1f of which %.1f in %llu address-based builds)", profile.bindingsUs / 1000.0, profile.uploadUs / 1000.0, profile.descriptorsUs / 1000.0, profile.otherUs / 1000.0, profile.addressOtherUs / 1000.0, static_cast<unsigned long long>(profile.addressBuilds));
        if (i == PhaseReadTarget && room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " (%llu resident lookups, %llu of them >= 1 ms = %.1f; target lookups: full-scissor %llu / partial %llu / pages walked %llu / %.0f us)", static_cast<unsigned long long>(profile.targetLookups), static_cast<unsigned long long>(profile.slowLookups), profile.slowLookupUs / 1000.0, static_cast<unsigned long long>(profile.fullScissorLookups), static_cast<unsigned long long>(profile.partialLookups), static_cast<unsigned long long>(profile.pagesWalked), profile.lookupUs);
    }
    if (room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "; per kind (avg us, count):");
    for (std::size_t kind = 0; kind < KindCount && room(); ++kind) {
        n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s %.0f (%llu)", DrawKindNames[kind], average(profile.kindUs[kind], profile.kindCounts[kind]), static_cast<unsigned long long>(profile.kindCounts[kind]));
    }
    // The longest draw and the longest single phase of any draw (which phase a long 'draw' hold
    // was), plus the hook waits nested inside the draws.
    if (room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "; longest draw %.1f ms, longest phase of any draw:", profile.maxDrawUs / 1000.0);
    for (std::size_t i = 0; i < PhaseCount && room(); ++i) {
        if (profile.maxUs[i] < 500.0) continue;
        n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s %.1f", DrawPhaseNames[i], profile.maxUs[i] / 1000.0);
    }
    if (room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "; hook waits inside draws %.1f ms (max %.1f; the draws' own syncs, %.1f ms, not counted)", profile.hookWaitUs / 1000.0, profile.maxHookWaitUs / 1000.0, profile.ownSyncUs / 1000.0);
    std::uint64_t indirectCpu = 0;
    for (std::size_t i = 1; i < IndirectPathCount; ++i) indirectCpu += profile.indirect[i];
    if ((profile.indirect[0] != 0 || indirectCpu != 0 || profile.indirectEmpty != 0) && room()) {
        n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "; indirect: gpu-side %llu (%llu rewritten), cpu-side %llu (", static_cast<unsigned long long>(profile.indirect[0]), static_cast<unsigned long long>(profile.indirectRewritten), static_cast<unsigned long long>(indirectCpu));
        for (std::size_t i = 1; i < IndirectPathCount && room(); ++i) {
            if (profile.indirect[i] != 0) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s %llu", IndirectDrawPathNames[i], static_cast<unsigned long long>(profile.indirect[i]));
        }
        if (room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "), empty skipped before their stages %llu, argument reads %.1f ms", static_cast<unsigned long long>(profile.indirectEmpty), profile.indirectReadUs / 1000.0);
    }
    if (room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "; skipped packets:");
    for (std::size_t i = 0; i < profile.skips.size() && room(); ++i) {
        n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s %llu in %.1f ms", DrawSkipNames[i], static_cast<unsigned long long>(profile.skips[i]), profile.skipUs[i] / 1000.0);
    }
    if (room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "; recipe hits %llu, misses by reason:", static_cast<unsigned long long>(profile.recipeHits));
    for (std::size_t i = 1; i < profile.recipeMisses.size() && room(); ++i) {
        n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s %llu", DrawRecipeMissName(static_cast<DrawRecipeMiss>(i)), static_cast<unsigned long long>(profile.recipeMisses[i]));
    }
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", line);
    AgcDriver::ProfilePrint_nid_no_patch("[rescache] draws: %llu hits, %llu misses, %llu invalidated, %llu uncacheable; validation memo %llu hits / %llu misses (which key words the misses differ in: the miss churn line)\n", static_cast<unsigned long long>(profile.cacheHits), static_cast<unsigned long long>(profile.cacheMisses), static_cast<unsigned long long>(profile.cacheInvalidated), static_cast<unsigned long long>(profile.uncacheable), static_cast<unsigned long long>(profile.validateHits), static_cast<unsigned long long>(profile.validateMisses));
    const auto address = AddressDrawReuseCounters();
    if (address != profile.addressSeen) {
        char reuse[512];
        const auto seen = [&](AddressDrawReuse outcome) { return static_cast<unsigned long long>(address[static_cast<std::size_t>(outcome)] - profile.addressSeen[static_cast<std::size_t>(outcome)]); };
        int at = std::snprintf(reuse, sizeof(reuse), "[rescache] address-based draws (10 s): reused %llu, built %llu; refused:", seen(AddressDrawReuse::Reused), seen(AddressDrawReuse::Built));
        for (auto outcome = static_cast<std::size_t>(AddressDrawReuse::NotReadOnly); outcome < address.size() && at > 0 && static_cast<std::size_t>(at) < sizeof(reuse); ++outcome) {
            at += std::snprintf(reuse + at, sizeof(reuse) - static_cast<std::size_t>(at), "%s %s %llu", outcome == static_cast<std::size_t>(AddressDrawReuse::NotReadOnly) ? "" : ",", AddressDrawReuseNames[outcome], seen(static_cast<AddressDrawReuse>(outcome)));
        }
        AgcDriver::ProfilePrint_nid_no_patch("%s\n", reuse);
        profile.addressSeen = address;
    }
    profile.totalsUs.fill(0);
    profile.maxUs.fill(0);
    profile.maxDrawUs = profile.hookWaitUs = profile.maxHookWaitUs = profile.ownSyncUs = 0;
    profile.bindingsUs = profile.uploadUs = profile.descriptorsUs = profile.otherUs = profile.addressOtherUs = 0;
    profile.addressBuilds = 0;
    profile.draws = profile.recorded = profile.waited = profile.completion = 0;
    profile.passesBegun = profile.passesContinued = 0;
    profile.reportedPassBreaks = profile.passBreaks;
    profile.reasons.fill(0);
    profile.recordedUs = profile.waitedUs = profile.synchronousUs = 0;
    profile.targetLookups = profile.slowLookups = 0;
    profile.slowLookupUs = 0;
    profile.fullScissorLookups = profile.partialLookups = profile.pagesWalked = 0;
    profile.lookupUs = 0;
    profile.kindUs.fill(0);
    profile.kindCounts.fill(0);
    profile.cacheHits = profile.cacheMisses = profile.cacheInvalidated = profile.uncacheable = 0;
    profile.validateHits = profile.validateMisses = 0;
    profile.recipeHits = 0;
    profile.recipeMisses.fill(0);
    profile.indirect.fill(0);
    profile.indirectRewritten = 0;
    profile.indirectReadUs = 0;
    profile.skips.fill(0);
    profile.skipUs.fill(0);
}

void countCache(std::uint64_t DrawProfile::*counter) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& stats = Profile();
    std::lock_guard lock(stats.mutex);
    ++(stats.*counter);
}

}

// ValidateShaders decodes every SPIR-V instruction of every stage on every draw. Its outcome depends
// only on the stages' compiled variants (a variant id names identical SPIR-V and binding layout),
// their push constant placement and vertex attribute shapes, and the state fields it checks, so the
// fragment output set is remembered by exactly those. Nothing here is keyed by pointer: the results
// a draw hands in live in a per-draw vector. A stage without a variant id is validated as before,
// except the rect-list control and evaluation stages, which are generated from the vertex and
// fragment results the key already names (the pipeline key treats them the same way). `memoized`
// says whether the memo applied, `hit` whether it answered. APS5_NO_VALIDATE_CACHE=1 validates
// every draw.
bool ValidationKey(const Context& context, std::span<const CompiledShader> shaders, const State& state, std::vector<std::uint64_t>& key) {
    using Stage = ShaderRecompiler::ShaderStage;
    static const bool disabled = std::getenv("APS5_NO_VALIDATE_CACHE") != nullptr;
    const auto add = [&](auto value) { key.push_back(static_cast<std::uint64_t>(value)); };
    bool keyed = !disabled;
    if (keyed) {
        key.reserve(40 + shaders.size() * 12);
        add(shaders.size());
        const bool patches = BarycentricPatches(state, shaders);
        add(patches ? static_cast<std::uint64_t>(state.provokingVertexMode) + 1u : 0u);
        for (const auto& shader : shaders) {
            Require(shader.program != nullptr, "missing compiled shader");
            const auto& program = *shader.program;
            const bool generated = ((state.rectList || patches) && (shader.stage == Stage::TessellationControl || shader.stage == Stage::TessellationEvaluation)) || shader.stage == Stage::Geometry;
            if (!generated && program.PipelineVariantId() == 0) {
                keyed = false;
                break;
            }
            add(shader.stage);
            add(generated ? std::uint64_t{0} : program.PipelineVariantId());
            add(KeyedPushOffset(shader));
            add(program.pushConstants.size());
            add(program.bdaAbiVersion);
            const auto& attributes = DrawVertexAttributes(shader);
            add(attributes.size());
            for (const auto& attribute : attributes) {
                add(attribute.location);
                add(attribute.components);
                add(attribute.fetchIndex);
                add(attribute.formatComponents);
                // The data format bits of the V# decide the attribute's signature.
                add((attribute.resource.fields[3] >> 12u) & 0x7fu);
            }
        }
    }
    if (keyed) {
        add(state.stages.path);
        add(state.rectList);
        add(state.topology);
        add(state.cullMode);
        add(state.polygonMode);
        add(state.lineWidth);
        add(state.blends.size());
        add(context.subgroup.subgroupSize);
        add(context.subgroup.supportedStages);
        add(context.subgroup.supportedOperations);
        add(context.fragmentShaderBarycentric);
        add(context.shaderClipDistance);
        add(context.shaderCullDistance);
        add(context.viewportIndexLayer);
        add(context.shaderTerminateInvocation);
        add(state.paClVsOutCntl);
        add(state.vertexPoints);
        add(state.stages.mesh.has_value());
        if (state.stages.mesh) {
            const auto& mesh = *state.stages.mesh;
            add(mesh.inputPrimitive);
            add(mesh.primitivesPerGroup);
            add(mesh.verticesPerGroup);
            add(mesh.maxVertices);
            add(mesh.maxPrimitives);
            add(mesh.threadsPerGroup);
            add(mesh.ldsSizeDwords);
            add(mesh.provokingVertex);
            add(mesh.outputPrimitiveSize);
        }
        add(state.stages.tessellation.has_value());
        if (state.stages.tessellation) {
            const auto& tessellation = *state.stages.tessellation;
            add(tessellation.inputControlPoints);
            add(tessellation.outputControlPoints);
            add(tessellation.domain);
            add(tessellation.partitioning);
            add(tessellation.outputTopology);
            add(tessellation.patchesPerGroup);
            add(tessellation.primitiveGroupSize);
            add(tessellation.breakAtInstance ? 1u : 0u);
        }
    }
    return keyed;
}

namespace {

AgcDriver::Mutex& validationMutex() {
    static AgcDriver::Mutex mutex;
    return mutex;
}

std::map<std::vector<std::uint64_t>, std::string>& validationFailures() {
    static std::map<std::vector<std::uint64_t>, std::string> failures;
    return failures;
}

std::map<std::vector<std::uint64_t>, std::uint32_t>& validationOutputs() {
    static std::map<std::vector<std::uint64_t>, std::uint32_t> outputs;
    return outputs;
}

std::vector<std::uint64_t>& validationKeyScratch() {
    struct KeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, KeyStorage>();
    key.clear();
    return key;
}

struct LastFragmentOutputs {
    std::vector<std::uint64_t> key;
    std::uint32_t outputs = 0;
    bool valid = false;
    FragmentOutputLookups lookups;
};

struct LastFragmentOutputsTag {};

bool LastFragmentOutputsEnabled() {
    static const bool enabled = std::getenv("APS5_NO_LAST_FRAGMENT_OUTPUTS") == nullptr;
    return enabled;
}

std::uint32_t CachedFragmentOutputs(const Context& context, std::span<const CompiledShader> shaders, const State& state, bool& memoized, bool& hit) {
    memoized = false;
    hit = false;
    auto& key = validationKeyScratch();
    const bool keyed = ValidationKey(context, shaders, state, key);
    auto& memo = validationOutputs();
    auto& last = ThreadScratch<LastFragmentOutputs, LastFragmentOutputsTag>();
    const bool remembers = keyed && LastFragmentOutputsEnabled();
    if (remembers && last.valid && key == last.key) {
        memoized = true;
        hit = true;
        ++last.lookups.last;
        return last.outputs;
    }
    const auto remember = [&](std::uint32_t outputs) {
        if (!remembers) return;
        last.key = key;
        last.outputs = outputs;
        last.valid = true;
    };
    if (keyed) {
        memoized = true;
        std::optional<std::uint32_t> found;
        {
            std::lock_guard lock(validationMutex());
            if (const auto entry = memo.find(key); entry != memo.end()) found = entry->second;
        }
        if (found) {
            hit = true;
            ++last.lookups.memo;
            remember(*found);
            return *found;
        }
    }
    ++last.lookups.validated;
    std::uint32_t outputs = 0;
    try {
        for (const auto location : ValidateShaders(shaders, state, context.subgroup, context.fragmentShaderBarycentric, context.descriptorIndexing, context.imageInt64Atomics, context.geometryShader, context.sampleRateShading, context.bufferInt64Atomics, context.shaderClipDistance, context.shaderCullDistance, context.viewportIndexLayer, context.runtimeDescriptorArray, context.demoteToHelperInvocation, context.shaderTerminateInvocation, context.meshShader, context.shaderResourceMinLod, context.fmaFloat16)) {
            Require(location < 32, "fragment output location exceeds the output mask");
            outputs |= 1u << location;
        }
    } catch (const std::exception& error) {
        if (keyed) {
            std::lock_guard lock(validationMutex());
            auto& failures = validationFailures();
            if (failures.size() >= 1024) failures.clear();
            failures.emplace(key, error.what());
        }
        throw;
    }
    if (keyed) {
        std::lock_guard lock(validationMutex());
        // A handful of configurations recur; a runaway key space is dropped wholesale.
        if (memo.size() >= 1024) memo.clear();
        memo.emplace(key, outputs);
    }
    remember(outputs);
    return outputs;
}

// The resource cache key of a recorded draw: a marker no compute key starts with (those begin with
// the stage), the device handle (the cache is process-wide and the driver replaces the headless
// device with the windowed one while workers may still use the old one: an entry's descriptor set
// and pooled buffers belong to the device that built it) and every stage's content key
// (ResourceCache::noteMiss walks this layout to attribute misses, so it changes together with it).
// With `ranges`, also the render target and the index buffer the build's alias checks compared the
// guest buffers against; without it the checks are repeated for a hit (CheckBufferAliases), since
// neither range is part of the descriptor set (the target is attached, the index buffer copied per
// draw), so a target or index ring that moves per frame does not miss on every draw. That guards a
// workload with such rings; in the profiled menu stage every draw was DRAW_INDEX_AUTO (index range
// 0) onto one fixed target, so its misses come from the stages' descriptor words themselves.
bool MovableBuffers() {
    static const bool enabled = std::getenv("APS5_NO_MOVED_BUFFER_TEMPLATES") == nullptr;
    return enabled;
}

ResourceCache::Key DrawResourceKey(const Context& context, std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::uint64_t indexBytes, bool ranges) {
    ResourceCache::Key key;
    key.reserve(256);
    key.push_back(0xffffffffu);
    const auto append64 = [&](std::uint64_t value) {
        key.push_back(static_cast<std::uint32_t>(value));
        key.push_back(static_cast<std::uint32_t>(value >> 32u));
    };
    append64(reinterpret_cast<std::uint64_t>(context.device));
    key.push_back(static_cast<std::uint32_t>(shaders.size()));
    for (const auto& shader : shaders) {
        const auto sizeAt = key.size();
        key.push_back(0u);
        auto& memo = shader.program->drawKeyMemo.words;
        if (auto words = memo.load(std::memory_order_acquire); words != nullptr && shader.heap == nullptr) {
            key.insert(key.end(), words->begin(), words->end());
        } else {
            ShaderResources::AppendContentKey(shader, true, MovableBuffers(), key);
            if (shader.heap == nullptr) memo.store(std::make_shared<const std::vector<std::uint32_t>>(key.begin() + static_cast<std::ptrdiff_t>(sizeAt + 1u), key.end()), std::memory_order_release);
        }
        key[sizeAt] = static_cast<std::uint32_t>(key.size() - sizeAt - 1u);
    }
    key.push_back(colorAttachments);
    if (ranges) {
        append64(target.address | target.pipeBankXor);
        append64(target.bytes);
        append64(indexAddress);
        append64(indexBytes);
    }
    return key;
}

// The alias checks the build makes for every guest buffer descriptor (ShaderResources::
// addGuestBuffer), repeated for a cached build whose own checks compared the buffers against
// another draw's render target and index buffer. The same conditions and messages, so a draw that
// would have failed its build fails its hit.
void CheckBufferAliases(std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint64_t indexAddress, std::uint64_t indexBytes) {
    const auto overlap = [](std::uint64_t first, std::uint64_t firstSize, std::uint64_t second, std::uint64_t secondSize) { return first < second + secondSize && second < first + firstSize; };
    const auto targetAddress = target.MemoryAddress();
    for (const auto& shader : shaders) {
        for (const auto& binding : shader.program->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers) continue;
            const auto& words = DrawGuestWords(shader, binding);
            for (std::size_t offset = 0; offset + 4 <= words.size(); offset += 4) {
                const ShaderRecompiler::ShaderBufferResource descriptor{{words[offset], words[offset + 1], words[offset + 2], words[offset + 3]}};
                const auto address = descriptor.Base48();
                auto size = descriptor.GetSize();
                if (size == 0 || address == 0) continue;
                const auto element = offset / 4;
                const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                if (!written) size = ShaderResources::BytesUpToTarget(address, size, descriptor.Stride(), &target);
                Require(!overlap(address, size, targetAddress, target.bytes), "shader buffer aliases the render target");
                Require(!written || !overlap(address, size, indexAddress, indexBytes), "writable shader buffer aliases the index buffer");
            }
        }
    }
}

}

std::array<std::uint32_t, 4> MeshIndexBufferDescriptor(const Pm4::DrawParameters& draw) {
    constexpr std::uint32_t RawWord3 = 0x31016facu;
    if (!draw.indexed) return {0u, 0u, 0u, RawWord3};
    const auto address = draw.indexAddress;
    const auto bytes = (static_cast<std::uint64_t>(draw.indexCount) * draw.indexSize + 3u) & ~std::uint64_t{3};
    Require(address != 0 && bytes != 0 && bytes <= 0xffffffffu && (address >> 48u) == 0, "invalid mesh index buffer range");
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) & 0xffffu, static_cast<std::uint32_t>(bytes), RawWord3};
}

MeshArgumentRules MeshArgumentRulesFor(const Context& context, const ShaderRecompiler::MeshConfiguration& mesh, std::uint32_t indexCount) {
    const auto inputSize = mesh.inputPrimitive == 1 ? 1u : mesh.inputPrimitive == 2 ? 2u : 3u;
    const auto step = mesh.inputPrimitive == 5 || mesh.inputPrimitive == 6 ? 1u : inputSize;
    Require(mesh.primitivesPerGroup != 0, "mesh pipeline groups no primitive (zero primitives per group)");
    return {indexCount, inputSize, step, mesh.primitivesPerGroup, context.meshLimits.maxMeshWorkGroupCount[0], context.meshLimits.maxMeshWorkGroupCount[1], context.meshLimits.maxMeshWorkGroupTotalCount};
}

std::optional<MeshGrid> MeshGridFor(const Context& context, std::uint32_t groups, std::uint32_t instances) {
    const auto& limits = context.meshLimits;
    if (groups == 0 || instances > limits.maxMeshWorkGroupCount[1] || limits.maxMeshWorkGroupCount[0] == 0) return std::nullopt;
    const auto across = groups <= limits.maxMeshWorkGroupCount[0] ? groups : std::min(limits.maxMeshWorkGroupCount[0], ShaderRecompiler::MeshLayerStrideLimit);
    const auto layers = (groups - 1u) / across + 1u;
    const auto columns = (groups - 1u) / layers + 1u;
    if (layers > limits.maxMeshWorkGroupCount[2] || (instances != 0 && columns > limits.maxMeshWorkGroupTotalCount / instances / layers)) return std::nullopt;
    return MeshGrid{columns, instances, layers, layers > 1u ? columns : 0u};
}

MeshArguments ResolveMeshArguments(const Pm4::DrawArguments& record, const MeshArgumentRules& rules) {
    const auto first = record.firstVertexOrIndex;
    const auto effective = first < rules.indexCount ? std::min(record.count, rules.indexCount - first) : 0u;
    std::uint32_t groups = 0;
    if (effective >= rules.inputSize && record.instances != 0) {
        groups = (effective - rules.inputSize) / rules.step / rules.primitivesPerGroup + 1u;
        if (groups > rules.maxGroups || record.instances > rules.maxInstances || groups > rules.maxTotal / record.instances) groups = 0;
    }
    return {groups, groups != 0 ? record.instances : 0u, groups != 0 ? 1u : 0u, effective, first};
}

std::shared_ptr<std::vector<std::shared_ptr<ShaderResources>>> DrawCopiedWriters() {
    // Never destroyed: a completion action of a batch still in flight at static teardown may erase from it.
    static auto* const writers = new std::shared_ptr<std::vector<std::shared_ptr<ShaderResources>>>(std::make_shared<std::vector<std::shared_ptr<ShaderResources>>>());
    return *writers;
}

namespace {
std::array<std::atomic<std::uint64_t>, 2> copiedWritersListed{};
std::atomic<std::uint64_t> idleStores{0};
std::atomic<std::uint64_t> templatesReplaced{0};
constexpr std::array<const char*, 3> FullDrawLookupNames{"template hit", "built", "not resolved"};
std::array<std::atomic<std::uint64_t>, FullDrawLookupNames.size()> fullDrawLookups{};
std::array<std::atomic<std::uint64_t>, FullDrawLookupNames.size()> fullDrawLookupNs{};
enum class FullDrawStep : std::uint8_t { Inputs, Targets, Depth, Lookup, Space, Proof, Moved, Checks, Pipeline, Bindings, Pass, Keep, Finish, Count };
constexpr std::size_t FullDrawStepCount = static_cast<std::size_t>(FullDrawStep::Count);
constexpr std::array<const char*, FullDrawStepCount> FullDrawStepNames{"inputs", "color targets", "depth target", "cache lookup", "address space", "proof", "moved buffers", "checks", "pipeline", "bindings", "pass and commands", "keep", "finish"};
constexpr std::size_t FullDrawCauseCount = static_cast<std::size_t>(FullDrawCause::Count);
constexpr std::array<const char*, FullDrawCauseCount> FullDrawCauseNames{"other", "today's path from a finisher", "today's path from the worker", "today's path inline", "record commit without a recipe", "record commit without a recipe, addressed stage", "record commit whose recipe missed", "record commit verify", "address-space recipe unused, switch off", "address-space recipe unused, content key moved", "address-space recipes refused after a verify difference", "address-space recipe verify"};
constexpr std::size_t RecordHeavyEventCount = static_cast<std::size_t>(RecordHeavyEvent::Count);

struct FullDrawSteps {
    std::chrono::steady_clock::time_point mark = std::chrono::steady_clock::now();
    std::array<std::uint64_t, FullDrawStepCount> ns{};
    void Lap(FullDrawStep step) {
        const auto now = std::chrono::steady_clock::now();
        ns[static_cast<std::size_t>(step)] += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - mark).count());
        mark = now;
    }
};

struct FullDrawStepCensus {
    using Counter = std::atomic<std::uint64_t>;
    std::array<Counter, FullDrawCauseCount> hits{};
    std::array<Counter, FullDrawCauseCount> hitNs{};
    std::array<Counter, FullDrawCauseCount> spaceHits{};
    std::array<Counter, FullDrawCauseCount> builds{};
    std::array<std::array<Counter, FullDrawStepCount>, FullDrawCauseCount> stepNs{};
};

FullDrawStepCensus& fullDrawStepCensus() {
    static auto* census = new FullDrawStepCensus();
    return *census;
}

struct RecordHeavyCensus {
    using Counter = std::atomic<std::uint64_t>;
    std::array<Counter, RecordHeavyEventCount> events{};
    Counter served{0};
    Counter servedNs{0};
    std::array<Counter, FullDrawStepCount> servedStepNs{};
    std::atomic<std::uint32_t> printed{0};
    Counter reportedPresent{0};
};

RecordHeavyCensus& recordHeavyCensus() {
    static auto* census = new RecordHeavyCensus();
    return *census;
}

bool fullDrawCensus() {
    static const bool census = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    return census;
}

struct FullDrawCauseStorage {};

FullDrawCause& fullDrawCause() {
    return HostThreadLocal<FullDrawCause, FullDrawCauseStorage>();
}

std::uint64_t addSteps(const FullDrawSteps& steps, std::array<std::atomic<std::uint64_t>, FullDrawStepCount>& into) {
    std::uint64_t total = 0;
    for (std::size_t step = 0; step < FullDrawStepCount; ++step) {
        total += steps.ns[step];
        if (steps.ns[step] != 0) into[step].fetch_add(steps.ns[step], std::memory_order_relaxed);
    }
    return total;
}

void noteFullDrawSteps(FullDrawCause cause, bool space, const FullDrawSteps& steps) {
    auto& census = fullDrawStepCensus();
    const auto index = std::min(static_cast<std::size_t>(cause), FullDrawCauseCount - 1u);
    const auto total = addSteps(steps, census.stepNs[index]);
    census.hits[index].fetch_add(1, std::memory_order_relaxed);
    census.hitNs[index].fetch_add(total, std::memory_order_relaxed);
    if (space) census.spaceHits[index].fetch_add(1, std::memory_order_relaxed);
}

void noteRecordHeavyServed(const FullDrawSteps& steps) {
    auto& census = recordHeavyCensus();
    const auto total = addSteps(steps, census.servedStepNs);
    census.served.fetch_add(1, std::memory_order_relaxed);
    census.servedNs.fetch_add(total, std::memory_order_relaxed);
}
}

namespace {
enum class TemplateBuildCause : std::uint8_t { NotKeyed, NoEntry, InUse, SpaceNotServed, Proof, NoRecorder, PatchedData, UnmovableBuffer, Count };
constexpr std::array<const char*, static_cast<std::size_t>(TemplateBuildCause::Count)> TemplateBuildCauseNames{"not keyed", "no entry under the key", "entry refused: address space in use", "entry refused: address space or snapshots not served", "entry refused: proof failed", "entry refused: no recorder", "entry refused: patched data", "entry refused: buffer not movable"};
enum class ProofSite : std::uint8_t { Lookup, Recipe, Record, Repeat, Count };
constexpr std::array<const char*, static_cast<std::size_t>(ProofSite::Count)> ProofSiteNames{"full lookup", "recipe", "record", "repeat"};
enum class ProofWriter : std::uint8_t { NoRange, Dispatch, HtileClear, Untagged, OtherWork, Recorded, Image, Depth, Unknown, Count };
constexpr std::array<const char*, static_cast<std::size_t>(ProofWriter::Count)> ProofWriterNames{"no range", "dispatch", "HTILE clear", "untagged recorded write", "copy/fill/clear", "recorded write (writer not kept)", "GPU image results pending", "depth surface", "no pending GPU write"};
constexpr std::array<const char*, static_cast<std::size_t>(ResourceCache::Fate::Count)> TemplateFateNames{"never noted", "built but not offered for insertion", "built but not recorded", "built but not reusable", "insert refused", "inserted then removed without a noted cause", "removed after a failed full-lookup proof", "removed: address space not served", "removed after a failed record proof", "capacity eviction", "table budget eviction", "dropped with a storage image", "dropped with a released texture"};
constexpr std::array<const char*, static_cast<std::size_t>(ShaderResources::NotReusable::Count)> TemplateNotReusableNames{"none", "HTILE clear", "copied writes", "lease or table", "multisample sources", "per-use textures", "data buffers", "no import serial"};
constexpr std::array<const char*, static_cast<std::size_t>(ContentKeyComponent::Count)> KeyPartNames{"layout", "V# word 0", "V# word 1", "V# word 2", "V# word 3", "image", "sampler", "shader-data", "other", "draw target/index"};
constexpr std::array<const char*, 4> RemovalAgeNames{"same present", "1 present back", "2-7 back", "8+ back"};
constexpr std::size_t ProofRefusals = static_cast<std::size_t>(ShaderResources::ProofRefusal::Count);
constexpr std::size_t FastFailSlots = static_cast<std::size_t>(ShaderResources::FastFail::Count) + 1u;
enum class NotKeyedReason : std::uint8_t { NoRecorder, RecordingOff, DumpLimit, TargetNotResident, CachesOff, RectListStage, GeometryStage, OtherStage, AddressStores, NeverReusable, Count };
constexpr std::array<const char*, static_cast<std::size_t>(NotKeyedReason::Count)> NotKeyedNames{"no recorder", "draws not recorded", "target dump", "target not resident", "template caches off", "rect-list generated stage", "geometry stage without a variant", "other stage without a variant", "address-based stores", "never reusable: other"};

struct TemplateKeepDrawCensus {
    using Counter = std::atomic<std::uint64_t>;
    std::array<Counter, static_cast<std::size_t>(NotKeyedReason::Count)> notKeyed{};
    Counter generatedHits{0};
    Counter generatedBuilt{0};
    Counter keptHits{0};
    Counter verified{0};
    Counter differing{0};
    Counter notRun{0};
};

TemplateKeepDrawCensus& templateKeepDrawCensus() {
    static auto* census = new TemplateKeepDrawCensus();
    return *census;
}

void noteNotKeyed(NotKeyedReason reason) {
    templateKeepDrawCensus().notKeyed[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

NotKeyedReason notKeyedReason(const State& state, std::span<const CompiledShader> shaders) {
    static const bool cachesOff = std::getenv("APS5_NO_DRAW_RESOURCE_CACHE") != nullptr || std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    if (cachesOff) return NotKeyedReason::CachesOff;
    using Stage = ShaderRecompiler::ShaderStage;
    for (const auto& shader : shaders) {
        if (shader.program != nullptr && shader.program->PipelineVariantId() != 0) continue;
        if (state.rectList && (shader.stage == Stage::TessellationControl || shader.stage == Stage::TessellationEvaluation)) return NotKeyedReason::RectListStage;
        return shader.stage == Stage::Geometry ? NotKeyedReason::GeometryStage : NotKeyedReason::OtherStage;
    }
    const bool stores = std::ranges::any_of(shaders, [](const CompiledShader& shader) { return shader.program != nullptr && shader.program->bdaStores && std::ranges::any_of(shader.program->bindings, [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable; }); });
    return stores ? NotKeyedReason::AddressStores : NotKeyedReason::NeverReusable;
}

std::uint64_t templateKeepsVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_TEMPLATE_KEEPS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

bool templateKeepsCensus() {
    return ResourceCache::FateCensus() || templateKeepsVerifyEvery() != 0;
}

struct ViewKeepDrawCensus {
    using Counter = std::atomic<std::uint64_t>;
    Counter revivedHits{0};
    Counter verified{0};
    Counter differing{0};
    Counter notRun{0};
};

ViewKeepDrawCensus& viewKeepDrawCensus() {
    static auto* census = new ViewKeepDrawCensus();
    return *census;
}

std::uint64_t viewTemplateKeepsVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_VIEW_TEMPLATE_KEEPS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

bool viewTemplateKeepsWatched() {
    return RuntimeSwitches::Set(RuntimeSwitch::ViewTemplateKeeps) && (ResourceCache::FateCensus() || viewTemplateKeepsVerifyEvery() != 0);
}

struct TemplateBuildCensus {
    using Counter = std::atomic<std::uint64_t>;
    std::array<Counter, TemplateBuildCauseNames.size()> builds{};
    std::array<Counter, TemplateFateNames.size()> fates{};
    std::array<Counter, TemplateNotReusableNames.size()> notReusable{};
    std::array<std::array<Counter, RemovalAgeNames.size()>, 2> removalAges{};
    std::array<std::array<Counter, ProofRefusals>, 2> removalRefusals{};
    Counter firstSeen{0};
    Counter sameKey{0};
    Counter otherKey{0};
    Counter otherKeyCached{0};
    std::array<Counter, KeyPartNames.size()> keyParts{};
    std::array<Counter, ProofSiteNames.size()> failedSites{};
    std::array<Counter, ProofSiteNames.size()> removedSites{};
    std::array<Counter, ProofRefusals> failed{};
    std::array<Counter, ProofRefusals> removed{};
    std::array<std::array<Counter, FastFailSlots>, ProofRefusals> fastFails{};
    std::array<std::array<Counter, ProofWriterNames.size()>, ProofRefusals> writers{};
    std::mutex mutex;
    std::unordered_map<std::uint64_t, ResourceCache::Key> lastKeys;
    std::uint64_t reportedPresent = 0;
};

TemplateBuildCensus& templateBuildCensus() {
    static auto* census = new TemplateBuildCensus();
    return *census;
}

std::uint64_t drawKeyShaders(const ResourceCache::Key& key) {
    std::uint64_t variants = 0;
    std::size_t at = 4;
    for (std::uint32_t stage = 0; stage < key[3] && at + 4 < key.size(); ++stage) {
        variants = variants * 1000003ull ^ (key[at + 3] | (static_cast<std::uint64_t>(key[at + 4]) << 32u));
        at += 1 + key[at];
    }
    return variants;
}

void noteTemplateLookup(const ResourceCache::Key& key, TemplateBuildCause cause, bool built) {
    auto& census = templateBuildCensus();
    ResourceCache::FateNote fate;
    if (built) {
        census.builds[static_cast<std::size_t>(cause)].fetch_add(1, std::memory_order_relaxed);
        if (cause == TemplateBuildCause::NotKeyed) return;
        if (cause == TemplateBuildCause::NoEntry) {
            fate = ResourceCache::FateOf(key);
            census.fates[static_cast<std::size_t>(fate.fate)].fetch_add(1, std::memory_order_relaxed);
            if (fate.fate == ResourceCache::Fate::LookupProof || fate.fate == ResourceCache::Fate::RecordProof) {
                const auto row = fate.fate == ResourceCache::Fate::LookupProof ? 0u : 1u;
                const auto present = Recorder::Presents();
                const auto age = present - std::min(present, fate.present);
                census.removalAges[row][age == 0 ? 0u : age == 1 ? 1u : age < 8 ? 2u : 3u].fetch_add(1, std::memory_order_relaxed);
                census.removalRefusals[row][std::min<std::size_t>(fate.detail, ProofRefusals - 1u)].fetch_add(1, std::memory_order_relaxed);
            } else if (fate.fate == ResourceCache::Fate::NotReusable) {
                census.notReusable[std::min<std::size_t>(fate.detail, census.notReusable.size() - 1u)].fetch_add(1, std::memory_order_relaxed);
            }
        }
        ResourceCache::NoteFate(key, ResourceCache::Fate::Built);
    }
    if (key.size() < 5 || key[0] != 0xffffffffu) return;
    const auto shaders = drawKeyShaders(key);
    std::lock_guard lock(census.mutex);
    if (census.lastKeys.size() >= 16384u) census.lastKeys.clear();
    auto& last = census.lastKeys[shaders];
    if (built && cause == TemplateBuildCause::NoEntry && fate.fate == ResourceCache::Fate::None) {
        if (last.empty()) {
            census.firstSeen.fetch_add(1, std::memory_order_relaxed);
        } else if (last == key) {
            census.sameKey.fetch_add(1, std::memory_order_relaxed);
        } else {
            census.otherKey.fetch_add(1, std::memory_order_relaxed);
            if (ResourceCache::FateOf(last).fate == ResourceCache::Fate::Inserted) census.otherKeyCached.fetch_add(1, std::memory_order_relaxed);
            ContentKeyDiff parts{};
            static_cast<void>(ShaderResources::DiffContentKeys(last, key, true, parts));
            for (std::size_t part = 0; part < parts.size(); ++part) {
                if (parts[part]) census.keyParts[part].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    last = key;
}

ProofWriter proofWriter(const Context& context, const ShaderResources::ProofReport& proof) {
    if (proof.address == 0) return ProofWriter::NoRange;
    const auto bytes = static_cast<std::size_t>(std::max<std::uint64_t>(proof.bytes, 1u));
    if (const auto* recorder = Recorder::Active(); recorder != nullptr && recorder->PendingWriteOverlaps(proof.address, bytes)) {
        if (!Recorder::KeyWriterCensus()) return ProofWriter::Recorded;
        const auto note = recorder->NewestWriteNote(proof.address, bytes);
        const auto tag = note != 0 ? Recorder::WriterOfNote(note) : ~std::uint64_t{0};
        if (tag == ~std::uint64_t{0}) return ProofWriter::Recorded;
        const auto site = tag >> 56u;
        return site == 1u ? ProofWriter::Dispatch : site == 8u ? ProofWriter::HtileClear : site == 0u ? ProofWriter::Untagged : ProofWriter::OtherWork;
    }
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> range{{{proof.address, proof.address + bytes}}};
    if (StorageTexture::AnyPendingOverlaps(range)) return ProofWriter::Image;
    if (DepthSurfaceOverlaps(context.device, proof.address, bytes)) return ProofWriter::Depth;
    return ProofWriter::Unknown;
}

void noteFailedProof(const Context& context, ProofSite site, const ShaderResources::ProofReport& proof, const ResourceCache::Key* removedKey) {
    if (!ResourceCache::FateCensus()) return;
    auto refusal = proof.refusal;
    if (refusal == ShaderResources::ProofRefusal::None) refusal = RuntimeSwitches::Set(RuntimeSwitch::XpProofAsIs) ? ShaderResources::ProofRefusal::AsIs : ShaderResources::ProofRefusal::Injected;
    const auto index = static_cast<std::size_t>(refusal);
    auto& census = templateBuildCensus();
    census.failedSites[static_cast<std::size_t>(site)].fetch_add(1, std::memory_order_relaxed);
    census.failed[index].fetch_add(1, std::memory_order_relaxed);
    census.fastFails[index][std::min<std::size_t>(proof.fastFail, FastFailSlots - 1u)].fetch_add(1, std::memory_order_relaxed);
    census.writers[index][static_cast<std::size_t>(proofWriter(context, proof))].fetch_add(1, std::memory_order_relaxed);
    if (removedKey == nullptr) return;
    census.removedSites[static_cast<std::size_t>(site)].fetch_add(1, std::memory_order_relaxed);
    census.removed[index].fetch_add(1, std::memory_order_relaxed);
    ResourceCache::NoteFate(*removedKey, site == ProofSite::Lookup ? ResourceCache::Fate::LookupProof : ResourceCache::Fate::RecordProof, static_cast<std::uint8_t>(refusal));
}
}

void NoteCopiedWriter(bool draw) {
    copiedWritersListed[draw ? 1u : 0u].fetch_add(1, std::memory_order_relaxed);
}

std::array<std::uint64_t, 2> CopiedWriterTotals() {
    return {copiedWritersListed[0].load(std::memory_order_relaxed), copiedWritersListed[1].load(std::memory_order_relaxed)};
}

void NoteIdleStore() {
    idleStores.fetch_add(1, std::memory_order_relaxed);
}

ResidentCensusTotals ResidentCensus() {
    ResidentCensusTotals totals;
    totals.cpuPublishes = CpuPublishesNoted();
    totals.idleStores = idleStores.load(std::memory_order_relaxed);
    totals.retires = HostImportRetires();
    totals.templatesReplaced = templatesReplaced.load(std::memory_order_relaxed);
    return totals;
}

std::string TakeFullDrawLookupText() {
    std::string text;
    for (std::size_t index = 0; index < FullDrawLookupNames.size(); ++index) {
        const auto count = fullDrawLookups[index].exchange(0, std::memory_order_relaxed);
        const auto ns = fullDrawLookupNs[index].exchange(0, std::memory_order_relaxed);
        if (count == 0) continue;
        char item[96];
        std::snprintf(item, sizeof(item), " %s %llu (%.1f us)", FullDrawLookupNames[index], static_cast<unsigned long long>(count), static_cast<double>(ns) / 1e3 / static_cast<double>(count));
        text += item;
    }
    return text;
}

FullDrawCauseScope::FullDrawCauseScope(FullDrawCause cause) {
    if (!fullDrawCensus()) return;
    auto& current = fullDrawCause();
    previous = current;
    current = cause;
    set = true;
}

FullDrawCauseScope::~FullDrawCauseScope() {
    if (set) fullDrawCause() = previous;
}

std::string TakeFullDrawStepText() {
    auto& census = fullDrawStepCensus();
    std::string text;
    for (std::size_t cause = 0; cause < FullDrawCauseCount; ++cause) {
        const auto hits = census.hits[cause].exchange(0, std::memory_order_relaxed);
        const auto ns = census.hitNs[cause].exchange(0, std::memory_order_relaxed);
        const auto space = census.spaceHits[cause].exchange(0, std::memory_order_relaxed);
        const auto builds = census.builds[cause].exchange(0, std::memory_order_relaxed);
        std::array<std::uint64_t, FullDrawStepCount> steps{};
        for (std::size_t step = 0; step < FullDrawStepCount; ++step) steps[step] = census.stepNs[cause][step].exchange(0, std::memory_order_relaxed);
        if (hits == 0 && builds == 0) continue;
        char item[192];
        std::snprintf(item, sizeof(item), "%s %s: template hits %llu (%.1f us", text.empty() ? "" : ";", FullDrawCauseNames[cause], static_cast<unsigned long long>(hits), hits == 0 ? 0.0 : static_cast<double>(ns) / 1e3 / static_cast<double>(hits));
        text += item;
        for (std::size_t step = 0; hits != 0 && step < FullDrawStepCount; ++step) {
            if (steps[step] == 0) continue;
            std::snprintf(item, sizeof(item), ", %s %.1f", FullDrawStepNames[step], static_cast<double>(steps[step]) / 1e3 / static_cast<double>(hits));
            text += item;
        }
        std::snprintf(item, sizeof(item), "; on an address-space template %llu), builds %llu", static_cast<unsigned long long>(space), static_cast<unsigned long long>(builds));
        text += item;
    }
    return text;
}

void NoteRecordHeavy(RecordHeavyEvent event) {
    recordHeavyCensus().events[std::min(static_cast<std::size_t>(event), RecordHeavyEventCount - 1u)].fetch_add(1, std::memory_order_relaxed);
}

void NoteRecordHeavyMiss(DrawRecipeMiss miss) {
    using enum RecordHeavyEvent;
    NoteRecordHeavy(miss == DrawRecipeMiss::NotRecordable ? MissNotRecordable : miss == DrawRecipeMiss::TargetGone ? MissTargets : miss == DrawRecipeMiss::TemplateGone ? MissTemplate : miss == DrawRecipeMiss::Proof ? MissProof : miss == DrawRecipeMiss::Moved ? MissMoved : MissObjects);
}

std::uint64_t RecordHeavyVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_RECORD_HEAVY_DRAWS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

void NoteRecordHeavyVerifyDifference(std::string_view field, std::string_view detail) {
    if (recordHeavyCensus().printed.fetch_add(1, std::memory_order_relaxed) < 8u) std::fprintf(stderr, "[record-heavy-draws-verify] %.*s differs: %.*s\n", static_cast<int>(field.size()), field.data(), static_cast<int>(detail.size()), detail.data());
}

std::string TakeRecordHeavyText() {
    auto& census = recordHeavyCensus();
    std::array<unsigned long long, RecordHeavyEventCount> events{};
    for (std::size_t index = 0; index < RecordHeavyEventCount; ++index) events[index] = census.events[index].exchange(0, std::memory_order_relaxed);
    const auto served = census.served.exchange(0, std::memory_order_relaxed);
    const auto servedNs = census.servedNs.exchange(0, std::memory_order_relaxed);
    std::string steps;
    for (std::size_t step = 0; step < FullDrawStepCount; ++step) {
        const auto ns = census.servedStepNs[step].exchange(0, std::memory_order_relaxed);
        if (ns == 0 || served == 0) continue;
        char item[64];
        std::snprintf(item, sizeof(item), ", %s %.1f", FullDrawStepNames[step], static_cast<double>(ns) / 1e3 / static_cast<double>(served));
        steps += item;
    }
    census.printed.store(0, std::memory_order_relaxed);
    const auto present = Recorder::Presents();
    const auto last = census.reportedPresent.exchange(present, std::memory_order_relaxed);
    const auto presents = present - std::min(present, last);
    const auto at = [&](RecordHeavyEvent event) { return events[static_cast<std::size_t>(event)]; };
    using enum RecordHeavyEvent;
    const auto missed = at(MissNotRecordable) + at(MissTargets) + at(MissTemplate) + at(MissObjects) + at(MissProof) + at(MissMoved);
    const auto refused = at(RefusedDropped) + at(RefusedNotCacheable) + at(RefusedWaited) + at(RefusedIndirect) + at(RefusedMoved) + at(RefusedMultisampled) + at(RefusedReorders) + at(RefusedShape);
    char text[1800];
    std::snprintf(text, sizeof(text), "%llu presents, APS5_RECORD_HEAVY_DRAWS %s: address-space templates committed from a record recipe %llu (%.1f us each%s); record route missed %llu (not recordable %llu, targets gone %llu, template gone or address space not served %llu [in use %llu, space or epoch change %llu, partial overlap %llu, snapshot changed %llu, snapshot outside the space %llu], objects gone %llu, proof %llu, moved buffers %llu); today's draw call instead: addressed record without a recipe %llu, recipe unused with the switch off %llu, content key moved %llu, refused after a verify difference %llu, verify %llu; address-space recipes built %llu, adopted %llu, not adopted %llu, refused at the build %llu (dropped early %llu, not cacheable %llu, waited %llu, indirect %llu, moved buffers %llu, multisampled %llu, reorders samples %llu, a shape the record route refuses %llu); verify %llu checked, %llu differing",
        static_cast<unsigned long long>(presents), RuntimeSwitches::Set(RuntimeSwitch::RecordHeavyDraws) ? "set" : "unset", static_cast<unsigned long long>(served), served == 0 ? 0.0 : static_cast<double>(servedNs) / 1e3 / static_cast<double>(served), steps.c_str(),
        missed, at(MissNotRecordable), at(MissTargets), at(MissTemplate), at(SpaceInUse), at(SpaceChanged), at(SpacePartial), at(SpaceSnapshotChanged), at(SpaceOutside), at(MissObjects), at(MissProof), at(MissMoved),
        at(FirstCommit), at(SwitchOff), at(KeyMoved), at(Refused), at(VerifyDue),
        at(Built), at(Adopted), at(NotAdopted), refused, at(RefusedDropped), at(RefusedNotCacheable), at(RefusedWaited), at(RefusedIndirect), at(RefusedMoved), at(RefusedMultisampled), at(RefusedReorders), at(RefusedShape),
        at(VerifyChecked), at(VerifyDiffering));
    return text;
}

std::string TakeTemplateBuildText() {
    auto& census = templateBuildCensus();
    const auto take = [](std::atomic<std::uint64_t>& counter) { return counter.exchange(0, std::memory_order_relaxed); };
    const auto list = [&](auto& counters, auto name) {
        std::string text;
        for (std::size_t index = 0; index < counters.size(); ++index) {
            const auto count = take(counters[index]);
            if (count == 0) continue;
            if (!text.empty()) text += ", ";
            text += name(index);
            text += ' ';
            text += std::to_string(count);
        }
        return text;
    };
    const auto refusalName = [](std::size_t index) { return ShaderResources::ProofRefusalName(static_cast<ShaderResources::ProofRefusal>(index)); };
    const auto fateDetail = [&](std::size_t index) {
        const auto fate = static_cast<ResourceCache::Fate>(index);
        std::string detail;
        if (fate == ResourceCache::Fate::None) {
            const auto first = take(census.firstSeen);
            const auto same = take(census.sameKey);
            const auto other = take(census.otherKey);
            const auto cached = take(census.otherKeyCached);
            const auto parts = list(census.keyParts, [](std::size_t part) { return KeyPartNames[part]; });
            detail = "shaders first seen " + std::to_string(first) + ", same key as the shaders' last lookup " + std::to_string(same) + ", same shaders under another key " + std::to_string(other) + " (that key still cached " + std::to_string(cached) + (parts.empty() ? std::string() : "; differing parts: " + parts) + ")";
        } else if (fate == ResourceCache::Fate::LookupProof || fate == ResourceCache::Fate::RecordProof) {
            const auto row = fate == ResourceCache::Fate::LookupProof ? 0u : 1u;
            detail = list(census.removalAges[row], [](std::size_t age) { return RemovalAgeNames[age]; });
            const auto refusals = list(census.removalRefusals[row], refusalName);
            if (!refusals.empty()) detail += "; refusals: " + refusals;
        } else if (fate == ResourceCache::Fate::NotReusable) {
            detail = list(census.notReusable, [](std::size_t reason) { return TemplateNotReusableNames[reason]; });
        }
        return detail;
    };
    std::string fates;
    for (std::size_t index = 0; index < census.fates.size(); ++index) {
        const auto count = take(census.fates[index]);
        const auto detail = fateDetail(index);
        if (count == 0) continue;
        if (!fates.empty()) fates += ", ";
        fates += TemplateFateNames[index];
        fates += ' ';
        fates += std::to_string(count);
        if (!detail.empty()) fates += " {" + detail + "}";
    }
    std::uint64_t built = 0;
    std::string causes;
    for (std::size_t index = 0; index < census.builds.size(); ++index) {
        const auto count = take(census.builds[index]);
        if (count == 0) continue;
        built += count;
        if (!causes.empty()) causes += ", ";
        causes += TemplateBuildCauseNames[index];
        causes += ' ';
        causes += std::to_string(count);
        if (static_cast<TemplateBuildCause>(index) == TemplateBuildCause::NoEntry && !fates.empty()) causes += " [" + fates + "]";
    }
    const auto refusalText = [&](auto& totals, bool details, std::uint64_t& sum) {
        std::string text;
        for (std::size_t index = 0; index < ProofRefusals; ++index) {
            const auto count = take(totals[index]);
            const auto fasts = details ? list(census.fastFails[index], [](std::size_t reason) { return ShaderResources::FastFailName(reason); }) : std::string();
            const auto writers = details ? list(census.writers[index], [](std::size_t writer) { return ProofWriterNames[writer]; }) : std::string();
            if (count == 0) continue;
            sum += count;
            if (!text.empty()) text += ", ";
            text += refusalName(index);
            text += ' ';
            text += std::to_string(count);
            if (details) text += " {fast fail: " + (fasts.empty() ? std::string("none") : fasts) + "; writer: " + (writers.empty() ? std::string("none") : writers) + "}";
        }
        return text;
    };
    std::uint64_t failed = 0;
    std::uint64_t removed = 0;
    const auto failedText = refusalText(census.failed, true, failed);
    const auto removedText = refusalText(census.removed, false, removed);
    const auto failedSites = list(census.failedSites, [](std::size_t site) { return ProofSiteNames[site]; });
    const auto removedSites = list(census.removedSites, [](std::size_t site) { return ProofSiteNames[site]; });
    const auto depth = TakeDepthCheckText();
    std::uint64_t presents = 0;
    {
        std::lock_guard lock(census.mutex);
        const auto present = Recorder::Presents();
        presents = present - std::min(present, census.reportedPresent);
        census.reportedPresent = present;
    }
    if (built == 0 && failed == 0 && removed == 0 && depth.empty()) return {};
    std::string text = std::to_string(presents) + " presents: built " + std::to_string(built);
    if (!causes.empty()) text += " (" + causes + ")";
    text += "; failed proofs " + std::to_string(failed);
    if (!failedSites.empty()) text += " (" + failedSites + ")";
    if (!failedText.empty()) text += ": " + failedText;
    text += "; removed after a failed proof " + std::to_string(removed);
    if (!removedSites.empty()) text += " (" + removedSites + ")";
    if (!removedText.empty()) text += ": " + removedText;
    if (!depth.empty()) text += "; multisample depth CheckCurrent: " + depth;
    return text;
}

std::string TakeTemplateKeepText() {
    auto& census = templateKeepDrawCensus();
    const auto take = [](std::atomic<std::uint64_t>& counter) { return counter.exchange(0, std::memory_order_relaxed); };
    std::string notKeyed;
    std::uint64_t notKeyedTotal = 0;
    for (std::size_t reason = 0; reason < census.notKeyed.size(); ++reason) {
        const auto count = take(census.notKeyed[reason]);
        if (count == 0) continue;
        notKeyedTotal += count;
        if (!notKeyed.empty()) notKeyed += ", ";
        notKeyed += std::string(NotKeyedNames[reason]) + " " + std::to_string(count);
    }
    const auto generatedHits = take(census.generatedHits);
    const auto generatedBuilt = take(census.generatedBuilt);
    const auto kept = take(census.keptHits);
    const auto verified = take(census.verified);
    const auto differing = take(census.differing);
    const auto notRun = take(census.notRun);
    const auto textures = ShaderResources::TakeTemplateKeepText();
    const bool on = RuntimeSwitches::Set(RuntimeSwitch::TemplateKeeps);
    std::string text = std::string("APS5_TEMPLATE_KEEPS ") + (on ? "set" : "unset") + "; template hits on a refilled texture " + std::to_string(kept);
    if (verified != 0 || notRun != 0) text += " (verified " + std::to_string(verified) + ", differing " + std::to_string(differing) + ", not run " + std::to_string(notRun) + ")";
    text += "; " + textures;
    text += "; not keyed " + std::to_string(notKeyedTotal) + (notKeyed.empty() ? std::string() : " (" + notKeyed + ")");
    text += "; keyed by the generated-stage rule: hits " + std::to_string(generatedHits) + ", built " + std::to_string(generatedBuilt);
    return text;
}

std::string TakeViewTemplateKeepText() {
    auto& census = viewKeepDrawCensus();
    const auto take = [](std::atomic<std::uint64_t>& counter) { return counter.exchange(0, std::memory_order_relaxed); };
    const auto hits = take(census.revivedHits);
    const auto verified = take(census.verified);
    const auto differing = take(census.differing);
    const auto notRun = take(census.notRun);
    const bool on = RuntimeSwitches::Set(RuntimeSwitch::ViewTemplateKeeps);
    std::string text = std::string("APS5_VIEW_TEMPLATE_KEEPS ") + (on ? "set" : "unset") + "; template hits on a taken-back texture " + std::to_string(hits);
    if (verified != 0 || notRun != 0) text += " (verified " + std::to_string(verified) + ", differing " + std::to_string(differing) + ", not run " + std::to_string(notRun) + ")";
    text += "; " + ShaderResources::TakeViewTemplateKeepText();
    return text;
}

const char* IndirectDrawPathName(IndirectDrawPath path) {
    return IndirectDrawPathNames[static_cast<std::size_t>(path)];
}

namespace {
std::array<std::atomic<std::uint64_t>, IndirectPathCount> indirectTotals{};
std::atomic<std::uint64_t> indirectReadTotalUs{0};
std::atomic<std::uint64_t> indirectRewrittenTotal{0};
std::atomic<std::uint64_t> indirectInPassTotal{0};

void countIndirectInPass() {
    indirectInPassTotal.fetch_add(1, std::memory_order_relaxed);
}
}

std::string IndirectDrawTotalsText() {
    std::string text = "gpu-side " + std::to_string(indirectTotals[0].load(std::memory_order_relaxed)) + " (" + std::to_string(indirectRewrittenTotal.load(std::memory_order_relaxed)) + " rewritten, " + std::to_string(indirectInPassTotal.load(std::memory_order_relaxed)) + " continued a pass); cpu-side:";
    for (std::size_t i = 1; i < IndirectPathCount; ++i) {
        const auto count = indirectTotals[i].load(std::memory_order_relaxed);
        if (count != 0) text += std::string(" ") + IndirectDrawPathNames[i] + " " + std::to_string(count);
    }
    return text + "; CPU argument reads " + std::to_string(indirectReadTotalUs.load(std::memory_order_relaxed) / 1000) + " ms";
}

void CountIndirectDraw(IndirectDrawPath path, double readMs, bool rewritten) {
    indirectTotals[static_cast<std::size_t>(path)].fetch_add(1, std::memory_order_relaxed);
    if (rewritten) indirectRewrittenTotal.fetch_add(1, std::memory_order_relaxed);
    indirectReadTotalUs.fetch_add(static_cast<std::uint64_t>(readMs * 1000.0), std::memory_order_relaxed);
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& stats = Profile();
    std::lock_guard lock(stats.mutex);
    ++stats.indirect[static_cast<std::size_t>(path)];
    if (rewritten) ++stats.indirectRewritten;
    stats.indirectReadUs += readMs * 1000.0;
}

std::array<std::uint64_t, IndirectPathCount> IndirectDrawCounts() {
    auto& stats = Profile();
    std::lock_guard lock(stats.mutex);
    return stats.indirect;
}

void CountEmptyIndirectDraw(double readMs) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& stats = Profile();
    std::lock_guard lock(stats.mutex);
    ++stats.indirectEmpty;
    stats.indirectReadUs += readMs * 1000.0;
}

std::uint64_t EmptyIndirectDrawCount() {
    auto& stats = Profile();
    std::lock_guard lock(stats.mutex);
    return stats.indirectEmpty;
}

std::array<DrawPassCounts, 2> DrawPassTotals() {
    auto& stats = Profile();
    std::lock_guard lock(stats.mutex);
    return stats.passTotals;
}

std::array<std::uint64_t, PassBreakCount> DrawPassBreaks() {
    auto& stats = Profile();
    std::lock_guard lock(stats.mutex);
    return stats.passBreaks;
}

void CountDrawSkip(DrawSkip kind, double us) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& stats = Profile();
    std::lock_guard lock(stats.mutex);
    ++stats.skips[static_cast<std::size_t>(kind)];
    stats.skipUs[static_cast<std::size_t>(kind)] += us;
}


bool DrawRecipes() {
    static const bool noDrawRecipe = std::getenv("APS5_NO_DRAW_RECIPE") != nullptr;
    return !noDrawRecipe;
}

const char* DrawRecipeMissName(DrawRecipeMiss miss) {
    constexpr std::array<const char*, static_cast<std::size_t>(DrawRecipeMiss::Count)> names{"none", "not recordable", "target gone", "template gone", "objects gone", "proof", "moved"};
    return names[static_cast<std::size_t>(miss)];
}

namespace {

// APS5_PROFILE_DRAW: the per-phase timers of one draw (DrawPhase), shared by Draw's parts and
// DrawWithRecipe.
struct DrawTimer {
    bool profile;
    std::chrono::steady_clock::time_point phaseStart;
    std::array<double, PhaseCount> us{};
    FullDrawSteps* steps = nullptr;
    explicit DrawTimer(bool profile) : profile(profile), phaseStart(std::chrono::steady_clock::now()) {}
    void lap(FullDrawStep step) {
        if (steps != nullptr) steps->Lap(step);
    }
    void phase(DrawPhase which) {
        if (!profile) return;
        const auto now = std::chrono::steady_clock::now();
        us[which] += std::chrono::duration<double, std::micro>(now - phaseStart).count();
        phaseStart = now;
    }
    void charge(DrawPhase which, std::chrono::steady_clock::time_point start) {
        if (!profile) return;
        const auto now = std::chrono::steady_clock::now();
        us[which] += std::chrono::duration<double, std::micro>(now - start).count();
        phaseStart += now - start;
    }
};

// One line per draw with the phases that took time (APS5_TRACE_DRAWS), and the 10 s totals.
// `waitedBefore` is the thread's GPU waits at the draw's start and `ownWaitedMs` the draw's own
// syncs (see DrawOutcome::hookWaitUs).
void reportDrawEnd(const State& state, const DrawTimer& timer, const ShaderResources::BuildTiming* built, DrawOutcome& outcome, double waitedBefore, double ownWaitedMs, const char* suffix) {
    if (!timer.profile) return;
    static const bool traceDraws = std::getenv("APS5_TRACE_DRAWS") != nullptr;
    outcome.hookWaitUs = std::max(0.0, Recorder::ThreadWaitedMs() - waitedBefore - ownWaitedMs) * 1000.0;
    outcome.ownSyncUs = ownWaitedMs * 1000.0;
    if (traceDraws) {
        char line[512];
        int n = std::snprintf(line, sizeof(line), "[draw] %ux%u %zu targets%s:", state.renderExtent.width, state.renderExtent.height, state.colors.size(), suffix);
        for (std::size_t i = 0; i < PhaseCount && n > 0 && static_cast<std::size_t>(n) < sizeof(line); ++i) {
            if (timer.us[i] <= 0) continue;
            n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s=%.0fus", DrawPhaseNames[i], timer.us[i]);
        }
        std::fprintf(stderr, "%s\n", line);
    }
    reportDraw(timer.us, built, outcome);
}

}

bool BindsIndexCopy(const State& state) {
    return !state.stages.mesh;
}

DrawInputCopy CopyDrawInput(const Context& context, Recorder* recorder, std::uint64_t address, std::size_t bytes, std::size_t alignment, Recorder::SnapshotUse use, std::size_t from, bool hostOnly) {
    Require(use != Recorder::SnapshotUse::Storage, "a draw input is a vertex or index buffer");
    Require(from == 0 || (use == Recorder::SnapshotUse::Vertex && from < bytes), "only a vertex input starts past its address");
    DrawInputCopy copy;
    const auto properties = use == Recorder::SnapshotUse::Vertex ? UploadMemoryProperties(context) : VkMemoryPropertyFlags{VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    copy.from = from;
    if (recorder != nullptr && bytes != 0) {
        GuestMemory::FlushGpuWrites(address, bytes);
#ifdef _WIN32
        const bool owned = !RuntimeSwitches::Set(RuntimeSwitch::NoOwnedDrawInputs);
#else
        constexpr bool owned = false;
#endif
        if (owned && recorder->OwnedDrawInputEntries() != 0) copy.buffer = recorder->ReusableOwnedDrawInput(address, bytes, use, &copy.derived, from, &copy.lowest);
        if (copy.buffer != nullptr && copy.buffer->Properties() != properties) copy.buffer = nullptr;
        if (copy.buffer != nullptr) {
            PerformanceTrace::CountOwnedDraw("OwnedDraw.input_hit");
            copy.reused = true;
            return copy;
        }
        copy.registryGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
        copy.generation = GuestMemory::CollectWrites(address, bytes);
        if (copy.generation != 0) copy.buffer = recorder->ReusableDrawSnapshot(address, bytes, use, &copy.derived, 0, nullptr, from, &copy.lowest, !hostOnly && (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0);
        else if (owned) copy.owned = recorder->ArmOwnedDrawInput(address, bytes);
        if (copy.buffer != nullptr && copy.buffer->Properties() != properties && copy.buffer->Mapped()) copy.buffer = nullptr;
        if (copy.buffer != nullptr) {
            PerformanceTrace::CountOwnedDraw("OwnedDraw.input_hit");
            copy.reused = true;
            return copy;
        }
    }
    PerformanceTrace::CountOwnedDraw("OwnedDraw.input_miss");
    copy.buffer = std::make_shared<Buffer>(context, bytes, (use == Recorder::SnapshotUse::Vertex ? VK_BUFFER_USAGE_VERTEX_BUFFER_BIT : VK_BUFFER_USAGE_INDEX_BUFFER_BIT) | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, properties);
    static const bool poisonPrefix = std::getenv("APS5_POISON_VERTEX_PREFIX") != nullptr;
    const auto held = copy.buffer->Bytes();
    if (poisonPrefix) std::fill(held.begin(), held.begin() + static_cast<std::ptrdiff_t>(from), std::byte{0x7f});
    GuestMemory::Read(address + from, held.subspan(from), alignment);
    if ((properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 && Recorder::GpuSideStats()) Recorder::CountGpuSide(Recorder::GpuSide::DeviceUploadBytes, bytes - from);
    return copy;
}

DrawInputCopy CopyZeroPaddedDrawInput(const Context& context, std::uint64_t address, std::size_t bytes, std::size_t validBytes, std::size_t stride) {
    Require(validBytes <= bytes, "the valid bytes of a vertex fetch exceed the fetch");
    auto readable = validBytes;
    if (stride != 0 && !GuestMemory::Accessible(reinterpret_cast<const void*>(address), validBytes)) {
        const auto ranges = GuestMemory::CommittedRanges(address, validBytes);
        const std::size_t prefix = !ranges.empty() && ranges.front().first == address ? static_cast<std::size_t>(ranges.front().second - address) : 0;
        if (validBytes - prefix < stride) readable = prefix;
    }
    if (readable != 0) GuestMemory::FlushGpuWrites(address, readable);
    DrawInputCopy copy;
    auto* recorder = context.recorder;
    const auto properties = UploadMemoryProperties(context);
    const bool keep = recorder != nullptr && readable != 0 && readable <= std::numeric_limits<std::uint32_t>::max();
    if (keep) {
        copy.registryGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
        copy.generation = GuestMemory::CollectWrites(address, readable);
        std::uint32_t keptReadable = 0;
        if (copy.generation != 0) copy.buffer = recorder->ReusableDrawSnapshot(address, bytes, Recorder::SnapshotUse::PaddedVertex, &keptReadable);
        if (copy.buffer != nullptr && keptReadable == readable && copy.buffer->Properties() == properties) {
            copy.reused = true;
            return copy;
        }
        copy.buffer = nullptr;
    }
    copy.buffer = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, properties);
    const auto span = copy.buffer->Bytes();
    GuestMemory::Read(address, span.subspan(0, readable), 1);
    std::fill(span.begin() + static_cast<std::ptrdiff_t>(readable), span.end(), std::byte{0});
    if ((properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 && Recorder::GpuSideStats()) Recorder::CountGpuSide(Recorder::GpuSide::DeviceUploadBytes, bytes);
    if (keep) recorder->KeepDrawSnapshot(address, bytes, copy.generation, copy.registryGeneration, copy.buffer, Recorder::SnapshotUse::PaddedVertex, static_cast<std::uint32_t>(readable));
    return copy;
}

void KeepDrawInput(Recorder* recorder, std::uint64_t address, const DrawInputCopy& copy, Recorder::SnapshotUse use, std::uint32_t derived, std::uint32_t lowest) {
    if (recorder == nullptr || copy.reused || copy.buffer == nullptr) return;
    if (copy.generation == 0) {
        if (!copy.owned.words.empty() && !RuntimeSwitches::Set(RuntimeSwitch::NoOwnedDrawInputs)) recorder->KeepOwnedDrawInput(address, copy.buffer->Bytes().size(), copy.owned, copy.buffer, use, derived, copy.from, lowest);
        return;
    }
    recorder->KeepDrawSnapshot(address, copy.buffer->Bytes().size(), copy.generation, copy.registryGeneration, copy.buffer, use, derived, copy.from, lowest);
}

// The last in-place answers of the thread (APS5_NO_INPUT_MEMO turns it off): the same range, alignment
// and kind in the same open batch answers as before while nothing that its checks read moved since
// they ran: no pending write, label or completion noted (Recorder::WriteGeneration), no store
// recorded or queued, the same pending-write snapshot, no storage result pending anew, the same
// host imports (a retired import moves them) and guest mappings. Its read is noted in the batch
// already. A range that went to a copy is not kept.
struct InPlaceInputMemo {
    struct Entry {
        const Recorder* recorder = nullptr;
        std::uint64_t open = 0, writes = 0, stores = 0, published = 0, pending = 0, imports = 0, mapping = 0;
        std::uint64_t address = 0;
        std::size_t bytes = 0, alignment = 0;
        bool vertex = false, behind = false;
        const HostImport* import = nullptr;
        bool Same(const Entry& other) const {
            return recorder == other.recorder && open == other.open && writes == other.writes && stores == other.stores && published == other.published && pending == other.pending && imports == other.imports && mapping == other.mapping && address == other.address && bytes == other.bytes && alignment == other.alignment && vertex == other.vertex;
        }
    };
    std::array<Entry, 32> entries{};
    std::size_t next = 0;
};
struct InPlaceInputMemoStorage {};

InPlaceInputMemo::Entry inPlaceInputState(const Recorder& recorder, std::uint64_t address, std::size_t bytes, std::size_t alignment, bool vertex) {
    InPlaceInputMemo::Entry state;
    state.recorder = &recorder;
    state.open = recorder.OpenSerial();
    state.writes = Recorder::WriteGeneration();
    state.stores = Recorder::QueuedStoreGeneration();
    state.published = Recorder::PublishGeneration();
    state.pending = StorageTexture::PendingSerial();
    state.imports = HostImportSetGeneration();
    state.mapping = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    state.address = address;
    state.bytes = bytes;
    state.alignment = alignment;
    state.vertex = vertex;
    return state;
}

namespace {

struct KnownImport {
    const HostImport* import = nullptr;
    std::uint64_t imports = 0;
    std::uint64_t mapping = 0;
    bool looked = false;
};

const HostImport* inPlaceImport(const Context& context, std::uint64_t address, std::size_t bytes, KnownImport* known) {
    if (known == nullptr) return HostImportFor(context, address, bytes);
    if (HostImportSetGeneration() == known->imports && GuestAllocations::GuestAllocationsGeneration_nid_postfix() == known->mapping) return known->import;
    known->looked = true;
    return HostImportFor(context, address, bytes);
}

const HostImport* inPlaceDrawInput(const Context& context, std::uint64_t address, std::size_t bytes, std::size_t alignment, bool* behindGpuWrite, KnownImport* known, std::size_t fetched = std::numeric_limits<std::size_t>::max(), std::uint64_t reads = std::numeric_limits<std::uint64_t>::max()) {
    auto* recorder = context.recorder;
    static const bool copyBehindGpuWrite = std::getenv("APS5_NO_BEHIND_INPUTS_IN_PLACE") != nullptr || std::getenv("APS5_NO_DRAW_INPUTS_IN_PLACE") != nullptr;
    const bool small = std::min(fetched, bytes) <= context.drawInputsInPlaceBelow && reads <= InPlaceDrawInputMaxReads;
    if (recorder == nullptr || bytes == 0 || (!context.drawInputsInPlace && !small && (behindGpuWrite == nullptr || copyBehindGpuWrite))) return nullptr;
    if (context.drawInputsInPlace && RuntimeSwitches::Set(RuntimeSwitch::XpCommitFloor)) {
        const auto* found = inPlaceImport(context, address, bytes, known);
        if (found == nullptr || HostImportOffset(*found, address) % alignment != 0) return nullptr;
        recorder->NotePendingRead(address, bytes, Recorder::ReadKind::DrawInput);
        return found;
    }
    const bool memo = context.drawInputsInPlace && !RuntimeSwitches::Set(RuntimeSwitch::NoInputMemo);
    // APS5_VERIFY_INPUT_MEMO: a memo hit runs the checks too, takes their answer and counts the hits
    // that answered otherwise ([input-memo-verify] on stderr).
    static const bool verify = std::getenv("APS5_VERIFY_INPUT_MEMO") != nullptr;
    auto& memoized = HostThreadLocal<InPlaceInputMemo, InPlaceInputMemoStorage>();
    const InPlaceInputMemo::Entry* remembered = nullptr;
    if (memo) {
        const auto now = inPlaceInputState(*recorder, address, bytes, alignment, behindGpuWrite != nullptr);
        for (const auto& entry : memoized.entries) {
            if (entry.import == nullptr || !entry.Same(now)) continue;
            remembered = &entry;
            break;
        }
        if (remembered != nullptr && !verify) {
            if (behindGpuWrite != nullptr && remembered->behind) *behindGpuWrite = true;
            return remembered->import;
        }
    }
    const auto rememberedImport = remembered != nullptr ? remembered->import : nullptr;
    const bool rememberedBehind = remembered != nullptr && remembered->behind;
    bool behind = false;
    const auto check = [&]() -> const HostImport* {
        if (behindGpuWrite == nullptr) {
            GuestMemory::FlushGpuWrites(address, bytes);
            if (recorder->PendingWriteOverlaps(address, bytes)) return nullptr;
        } else {
            StorageTexture::FlushPending(address, bytes, nullptr, "in-place vertex input");
            behind = recorder->PendingWriteOverlaps(address, bytes);
            if (!behind && !context.drawInputsInPlace && !small) return nullptr;
        }
        if (recorder->PendingLabelIn(address, bytes) || recorder->QueuedStoreOverlaps(address, bytes)) return nullptr;
        const auto overlaps = [&](const auto& writer) { return writer->WritesOverlap(address, bytes); };
        if (context.copiedWriters != nullptr && std::any_of(context.copiedWriters->begin(), context.copiedWriters->end(), overlaps)) return nullptr;
        const auto drawWriters = DrawCopiedWriters();
        if (std::any_of(drawWriters->begin(), drawWriters->end(), overlaps)) return nullptr;
        const auto* found = inPlaceImport(context, address, bytes, known);
        if (found == nullptr || HostImportOffset(*found, address) % alignment != 0) return nullptr;
        recorder->NotePendingRead(address, bytes, Recorder::ReadKind::DrawInput);
        return found;
    };
    const auto* import = check();
    if (verify && rememberedImport != nullptr) {
        static std::atomic<std::uint64_t> checks{0}, differences{0};
        const bool differed = import != rememberedImport || (behindGpuWrite != nullptr && behind != rememberedBehind);
        if (differed) differences.fetch_add(1, std::memory_order_relaxed);
        const auto count = checks.fetch_add(1, std::memory_order_relaxed) + 1u;
        if (differed || count % 262144u == 0u || (count & (count - 1u)) == 0u) std::fprintf(stderr, "[input-memo-verify] %llu checked, %llu differed\n", static_cast<unsigned long long>(count), static_cast<unsigned long long>(differences.load()));
    }
    if (import == nullptr) return nullptr;
    if (behindGpuWrite != nullptr && behind) *behindGpuWrite = true;
    if (memo) {
        // The state after the checks (their flushes may have moved it), so the next identical
        // draw finds it unchanged.
        auto entry = inPlaceInputState(*recorder, address, bytes, alignment, behindGpuWrite != nullptr);
        entry.behind = behind;
        entry.import = import;
        memoized.entries[memoized.next++ % memoized.entries.size()] = entry;
    }
    return import;
}

}

const HostImport* InPlaceDrawInput(const Context& context, std::uint64_t address, std::size_t bytes, std::size_t alignment, bool* behindGpuWrite, std::size_t fetched, std::uint64_t reads) {
    return inPlaceDrawInput(context, address, bytes, alignment, behindGpuWrite, nullptr, fetched, reads);
}

InPlaceIndices InPlaceIndexRange(Recorder& recorder, std::uint64_t address, std::size_t bytes, std::uint32_t indexSize, bool skipRestart) {
    const std::span<const std::byte> range(reinterpret_cast<const std::byte*>(address), bytes);
    InPlaceIndices indices;
    const auto scan = [&] {
        const auto bounds = DrawIndexRange(range, indexSize, skipRestart);
        if (!bounds) return;
        indices.highest = bounds->highest;
        indices.lowest = bounds->lowest;
    };
    if (bytes < Recorder::KeptIndexRangeIndices * indexSize) {
        scan();
        return indices;
    }
    const auto use = indexSize == 2 ? (skipRestart ? Recorder::SnapshotUse::Index16Restart : Recorder::SnapshotUse::Index16) : (skipRestart ? Recorder::SnapshotUse::Index32Restart : Recorder::SnapshotUse::Index32);
    const auto registryGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    const auto generation = GuestMemory::CollectWrites(address, bytes);
    if (generation != 0 && recorder.KnownHighestIndex(address, bytes, use, indices.highest, &indices.lowest)) {
        indices.reused = true;
        return indices;
    }
    scan();
    recorder.KeepHighestIndex(address, bytes, use, generation, registryGeneration, indices.highest, indices.lowest);
    return indices;
}

namespace {

// The draw's inputs before its resources (prepareDrawInputs): the validated parameters, the index
// buffer copy with its highest index, the vertex buffer copies and their layout, the fragment
// outputs and the pipeline stages.
struct DrawInputs {
    // An empty auto draw: nothing to record.
    bool nothing = false;
    std::uint64_t indexBytes = 0;
    std::shared_ptr<Buffer> indices;
    VkBuffer indexHandle = VK_NULL_HANDLE;
    VkDeviceSize indexOffset = 0;
    std::uint32_t maxIndex = 0;
    std::uint32_t minIndex = 0;
    std::optional<std::uint32_t> indexCount;
    std::uint32_t listRestartPrimitive = 0;
    VertexInputLayout vertexInput;
    std::shared_ptr<const VertexInputLayout> retainedVertexInput;
    std::vector<std::shared_ptr<Buffer>> vertexBuffers;
    std::vector<VkBuffer> vertexHandles;
    std::vector<VkDeviceSize> vertexOffsets;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> inPlaceReads;
    bool vertexInputBehindGpuWrite = false;
    std::uint32_t fragmentOutputs = 0;
    VkPipelineStageFlags shaderStages = 0;
    std::uint32_t meshGroups = 0;
    std::uint32_t meshLayers = 1;
    std::uint32_t meshLayerStride = 0;
    std::vector<std::uint32_t> restartTable;
    std::uint32_t expandedIndexCount = 0;
    bool recycled = false;
    const VertexInputLayout& VertexLayout() const { return retainedVertexInput != nullptr ? *retainedVertexInput : vertexInput; }
};

struct DrawInputSpare {
    std::vector<VkBuffer> vertexHandles;
    std::vector<VkDeviceSize> vertexOffsets;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> inPlaceReads;
};

struct DrawInputScratch {
    std::vector<VertexFetch> fetches;
    std::vector<std::size_t> fetchValid;
    std::vector<std::size_t> fetchOf;
    std::vector<std::shared_ptr<Buffer>> zeroed;
    std::vector<std::size_t> soloFetches;
    std::vector<bool> isSolo;
    std::vector<VertexFetch> plannedFetches;
    std::vector<std::size_t> plannedOf;
    VertexCopyPlan plan;
    std::vector<std::size_t> order;
    std::vector<VkBuffer> rangeHandles;
    std::vector<VkDeviceSize> rangeOffsets;
    std::vector<VkBuffer> soloHandles;
    bool busy = false;
};

void takeSpareInputs(DrawInputs& inputs) {
    auto& spare = HostThreadLocal<DrawInputSpare, DrawInputSpare>();
    inputs.vertexHandles = std::move(spare.vertexHandles);
    inputs.vertexOffsets = std::move(spare.vertexOffsets);
    inputs.inPlaceReads = std::move(spare.inPlaceReads);
    inputs.vertexHandles.clear();
    inputs.vertexOffsets.clear();
    inputs.inPlaceReads.clear();
    inputs.recycled = true;
}

void returnSpareInputs(DrawInputs& inputs) {
    if (!inputs.recycled) return;
    auto& spare = HostThreadLocal<DrawInputSpare, DrawInputSpare>();
    spare.vertexHandles = std::move(inputs.vertexHandles);
    spare.vertexOffsets = std::move(inputs.vertexOffsets);
    spare.inPlaceReads = std::move(inputs.inPlaceReads);
    inputs.recycled = false;
}

bool StripPatches(const State& state, std::span<const CompiledShader> shaders) {
    return state.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP && BarycentricPatches(state, shaders);
}

std::shared_ptr<Buffer> zeroVertexBuffer(const Context& context, std::size_t bytes) {
    auto zero = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    std::fill(zero->Bytes().begin(), zero->Bytes().end(), std::byte{0});
    return zero;
}

void requireUncountedKills(const State& state) {
    if (!state.pixelKill || state.pixelEarlyZ || !std::ranges::any_of(state.colors, [](const ColorTarget& color) { return color.partialWriteMask != 0; })) return;
    const auto* recorder = Recorder::Active();
    Require(recorder == nullptr || !recorder->CountingSamples(), "a partial write of a packed color target merges inside the pixel shader after early depth and stencil tests, which a shader that kills pixels under late tests cannot use while occlusion queries count samples: the early tests would count the pixels it kills");
}

// Draw's validation, index and vertex phases. With `recipe` the fragment outputs, the pipeline
// stages and the vertex input layout are the recipe's (derived from the same compiled stages)
// instead of computed.
struct DrawPlan {
    bool nothing = false;
    std::uint64_t indexBytes = 0;
    std::uint32_t fragmentOutputs = 0;
    VkPipelineStageFlags shaderStages = 0;
    std::uint32_t meshGroups = 0;
    std::uint32_t meshLayers = 1;
    std::uint32_t meshLayerStride = 0;
    VertexInputLayout vertexInput;
    std::shared_ptr<const VertexInputLayout> retainedVertexInput;
    bool candidate = false;
    bool keyed = false;
    bool generatedKey = false;
    ResourceCache::Key contentKey;
    const VertexInputLayout& VertexLayout() const { return retainedVertexInput != nullptr ? *retainedVertexInput : vertexInput; }
};

bool GeneratedRectStage(const CompiledShader& shader) {
    using Stage = ShaderRecompiler::ShaderStage;
    if (shader.program == nullptr || shader.heap != nullptr || shader.program->PipelineVariantId() != 0) return false;
    if (shader.stage != Stage::TessellationControl && shader.stage != Stage::TessellationEvaluation) return false;
    return std::ranges::all_of(shader.program->bindings, [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::FaultBuffer; });
}

bool EarlyDrawKeys() {
    static const bool enabled = std::getenv("APS5_NO_EARLY_DRAW_KEY") == nullptr;
    return enabled;
}

bool KeyForDevice(const std::vector<std::uint32_t>& key, VkDevice device) {
    const auto handle = reinterpret_cast<std::uint64_t>(device);
    return key.size() >= 3 && key[0] == 0xffffffffu && key[1] == static_cast<std::uint32_t>(handle) && key[2] == static_cast<std::uint32_t>(handle >> 32u);
}

DrawPlan planDraw(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, const DrawRecipe* recipe, DrawOutcome& outcome, DrawTimer& timer, const std::uint32_t* fragmentOutputs = nullptr, const std::vector<std::uint32_t>* earlyKey = nullptr) {
    DrawPlan plan;
    APS5_LOG_OUT_DEBUG("Draw indices=%u instances=%u indexSize=%u flags=%u indexAddress=0x%llx", draw.indexCount, draw.instanceCount, draw.indexSize, draw.flags, static_cast<unsigned long long>(draw.indexAddress));
    APS5_LOG_OUT_DEBUG("State colorTarget=%u render=%ux%u colorAddress=0x%llx colorBytes=%llu colorExtent=%ux%u", state.hasColorTarget ? 1u : 0u, state.renderExtent.width, state.renderExtent.height, static_cast<unsigned long long>(state.color.address), static_cast<unsigned long long>(state.color.bytes), state.color.extent.width, state.color.extent.height);
    APS5_LOG_OUT_DEBUG("Viewport x=%f y=%f w=%f h=%f minDepth=%f maxDepth=%f", state.viewport.x, state.viewport.y, state.viewport.width, state.viewport.height, state.viewport.minDepth, state.viewport.maxDepth);
    APS5_LOG_OUT_DEBUG("Scissor x=%d y=%d w=%u h=%u topology=%u cullMode=0x%x frontFace=%u", state.scissor.offset.x, state.scissor.offset.y, state.scissor.extent.width, state.scissor.extent.height, static_cast<unsigned>(state.topology), static_cast<unsigned>(state.cullMode), static_cast<unsigned>(state.frontFace));
    const auto* args = draw.indirect ? &*draw.indirect : nullptr;
    Require(draw.indexed ? draw.flags == 0 : (draw.flags & ~0x20u) == 0, "draw modifiers are unsupported");
    if (draw.indexed) {
        Require(draw.indexSize == 2 || draw.indexSize == 4, "only uint16 and uint32 index buffers are supported");
    } else {
        Require(draw.indexAddress == 0 && draw.indexSize == 0, "auto draw must not reference an index buffer");
        if (args == nullptr) {
            if (draw.indexCount == 0 || draw.instanceCount == 0) {
                plan.nothing = true;
                return plan;
            }
            Require(draw.firstVertex <= std::numeric_limits<std::uint32_t>::max() - (draw.indexCount - 1u), "auto draw vertex range overflow");
            Require(draw.firstInstance <= std::numeric_limits<std::uint32_t>::max() - (draw.instanceCount - 1u), "auto draw instance range overflow");
        }
    }
    if (args == nullptr) {
        if (draw.indexCount == 0 || draw.instanceCount == 0) {
            plan.nothing = true;
            return plan;
        }
    } else Require((!state.stages.mesh || draw.indexed) && !state.stages.tessellation && !state.quadList, "indirect draw on a non-vertex path must be resolved by the driver");
    plan.indexBytes = static_cast<std::uint64_t>(draw.indexCount) * draw.indexSize;
    const auto indexBytes = plan.indexBytes;
    APS5_LOG_OUT_DEBUG("Index buffer bytes=%llu", static_cast<unsigned long long>(indexBytes));
    Require(indexBytes <= std::numeric_limits<std::size_t>::max(), "index buffer size overflow");
    Require(!draw.indexed || !state.hasColorTarget || draw.indexAddress + indexBytes <= state.color.MemoryAddress() || state.color.MemoryAddress() + state.color.bytes <= draw.indexAddress, "index buffer aliases the render target");
    APS5_LOG_CHARS_OUT_DEBUG("ValidateShaders");
    if (recipe != nullptr) {
        plan.fragmentOutputs = recipe->fragmentOutputs;
        plan.shaderStages = recipe->shaderStages;
    } else if (fragmentOutputs != nullptr && RuntimeSwitches::Set(RuntimeSwitch::CarryValidation)) {
        plan.fragmentOutputs = *fragmentOutputs;
        plan.shaderStages = PipelineStages(shaders);
        outcome.validateMemoized = true;
        outcome.validateHit = true;
    } else {
        plan.fragmentOutputs = CachedFragmentOutputs(context, shaders, state, outcome.validateMemoized, outcome.validateHit);
        plan.shaderStages = PipelineStages(shaders);
    }
    APS5_LOG_CHARS_OUT_DEBUG("ValidateShaders OK");
    APS5_LOG_OUT_DEBUG("PipelineStages=0x%x", static_cast<unsigned>(plan.shaderStages));
    if (state.stages.mesh) {
        APS5_LOG_CHARS_OUT_DEBUG("Mesh path");
        Require(context.meshShader, "device does not support mesh shaders");
        const auto& mesh = *state.stages.mesh;
        const auto inputSize = mesh.inputPrimitive == 1 ? 1u : mesh.inputPrimitive == 2 ? 2u : 3u;
        Require(mesh.primitivesPerGroup != 0, "mesh pipeline groups no primitive (zero primitives per group)");
        Require(((state.stages.registerValue >> 19u) & 3u) == 0 || !draw.indexed, "indexed fast-launch draws are unsupported");
        if (args == nullptr && draw.indexCount < inputSize) {
            plan.nothing = true;
            return plan;
        }
        if (args == nullptr) {
            const auto step = mesh.inputPrimitive == 5 || mesh.inputPrimitive == 6 ? 1u : inputSize;
            const auto primitives = (draw.indexCount - inputSize) / step + 1u;
            const auto groups = (primitives - 1u) / mesh.primitivesPerGroup + 1u;
            const auto grid = MeshGridFor(context, groups, draw.instanceCount);
            Require(grid.has_value(), "mesh draw exceeds workgroup count limits");
            APS5_LOG_OUT_DEBUG("Mesh primitives=%u groups=%u as %u x %u", primitives, groups, grid->groups, grid->layers);
            plan.meshGroups = grid->groups;
            plan.meshLayers = grid->layers;
            plan.meshLayerStride = grid->layerStride;
        }
    }
    if (state.stages.tessellation) Require(draw.indexCount % state.stages.tessellation->inputControlPoints == 0, "incomplete tessellation patch");
    ValidateViewport(context, state.viewport);
    for (const auto& viewport : state.extraViewports) ValidateViewport(context, viewport);
    ValidateDepthBounds(context, state);
    ValidateSampleMask(context, state, shaders);
    timer.phase(PhaseValidate);
    const auto layoutStart = timer.profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto& attributes = DrawVertexAttributes(shaders.front());
    if (recipe != nullptr) {
        plan.vertexInput = recipe->vertexInput;
    } else if (!attributes.empty()) {
        if (context.vertexInputs != nullptr) plan.retainedVertexInput = context.vertexInputs->TryGet(shaders.front().program->variantId, attributes);
        if (plan.retainedVertexInput == nullptr) plan.vertexInput = BuildVertexInputLayout(context, attributes);
    }
    timer.charge(PhaseVertex, layoutStart);
    static const bool noDrawResourceCache = std::getenv("APS5_NO_DRAW_RESOURCE_CACHE") != nullptr;
    static const bool noTextureCache = std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    static const bool trimKey = std::getenv("APS5_NO_DRAW_KEY_TRIM") == nullptr;
    const bool generatedRule = recipe == nullptr && state.rectList && RuntimeSwitches::Set(RuntimeSwitch::TemplateKeeps) && !BarycentricPatches(state, shaders);
    plan.candidate = recipe == nullptr && !noDrawResourceCache && !noTextureCache && std::all_of(shaders.begin(), shaders.end(), [&](const CompiledShader& shader) { return shader.program != nullptr && (shader.program->PipelineVariantId() != 0 || (generatedRule && GeneratedRectStage(shader))); });
    plan.generatedKey = generatedRule && plan.candidate && std::ranges::any_of(shaders, [](const CompiledShader& shader) { return shader.program->PipelineVariantId() == 0; });
    plan.keyed = plan.candidate && !ShaderResources::NeverReusable(shaders);
    if (plan.keyed) {
        const auto keyStart = timer.profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (earlyKey != nullptr && KeyForDevice(*earlyKey, context.device)) {
            plan.contentKey = *earlyKey;
            static const bool verify = std::getenv("APS5_VERIFY_EARLY_DRAW_KEY") != nullptr;
            if (verify) Require(plan.contentKey == DrawResourceKey(context, shaders, state.color, static_cast<std::uint32_t>(state.colors.size()), draw.indexAddress, indexBytes, !trimKey), "the draw resource key made before the commit differs from the commit's");
        } else {
            plan.contentKey = DrawResourceKey(context, shaders, state.color, static_cast<std::uint32_t>(state.colors.size()), draw.indexAddress, indexBytes, !trimKey);
        }
        timer.charge(PhaseLookup, keyStart);
    }
    return plan;
}

bool VerifyDrawPlan() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_PLAN") != nullptr;
    return verify;
}

void requireDrawPlanRepeats(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, const DrawPlan& plan) {
    DrawOutcome scratch;
    DrawTimer scratchTimer(false);
    const auto again = planDraw(context, state, draw, shaders, nullptr, scratch, scratchTimer);
    const auto sameBinding = [](const VkVertexInputBindingDescription& a, const VkVertexInputBindingDescription& b) { return a.binding == b.binding && a.stride == b.stride && a.inputRate == b.inputRate; };
    const auto sameAttribute = [](const VkVertexInputAttributeDescription& a, const VkVertexInputAttributeDescription& b) { return a.location == b.location && a.binding == b.binding && a.format == b.format && a.offset == b.offset; };
    const auto& layout = plan.VertexLayout();
    const auto& repeated = again.VertexLayout();
    Require(again.nothing == plan.nothing && again.indexBytes == plan.indexBytes && again.fragmentOutputs == plan.fragmentOutputs && again.shaderStages == plan.shaderStages && again.meshGroups == plan.meshGroups && again.meshLayers == plan.meshLayers && again.meshLayerStride == plan.meshLayerStride && again.candidate == plan.candidate && again.keyed == plan.keyed && again.contentKey == plan.contentKey && std::ranges::equal(repeated.bindings, layout.bindings, sameBinding) && std::ranges::equal(repeated.attributes, layout.attributes, sameAttribute), "APS5_VERIFY_DRAW_PLAN: the plan built after the draw's record-time work differs from the plan the draw used");
}

DrawInputs prepareDrawInputs(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, const DrawPlan& drawPlan, DrawTimer& timer, PerformanceTimer& timing, const DrawRecipe* recipe = nullptr) {
    DrawInputs inputs;
    const auto reads = draw.indirect ? std::numeric_limits<std::uint64_t>::max() : static_cast<std::uint64_t>(draw.indexCount) * draw.instanceCount;
    inputs.indexBytes = drawPlan.indexBytes;
    inputs.fragmentOutputs = drawPlan.fragmentOutputs;
    inputs.shaderStages = drawPlan.shaderStages;
    inputs.meshGroups = drawPlan.meshGroups;
    inputs.meshLayers = drawPlan.meshLayers;
    inputs.meshLayerStride = drawPlan.meshLayerStride;
    inputs.vertexInput = drawPlan.vertexInput;
    inputs.retainedVertexInput = drawPlan.retainedVertexInput;
    const auto* args = draw.indirect ? &*draw.indirect : nullptr;
    const auto indexBytes = inputs.indexBytes;
    if (draw.indexed) GuestMemory::CheckRange(reinterpret_cast<const void*>(draw.indexAddress), static_cast<std::size_t>(indexBytes), draw.indexSize);
    APS5_LOG_CHARS_OUT_DEBUG("Index buffer range OK");
    if (draw.indexed) RequireNoMultisampleTarget(draw.indexAddress, static_cast<std::size_t>(indexBytes), "an index buffer read");
    if (state.pixelKill && !state.pixelEarlyZ && std::ranges::any_of(state.colors, [](const ColorTarget& color) { return color.partialWriteMask != 0; })) {
        const auto* recorder = Recorder::Active();
        Require(recorder == nullptr || !recorder->CountingSamples(), "a partial write of a packed color target merges inside the pixel shader after early depth and stencil tests, which a shader that kills pixels under late tests cannot use while occlusion queries count samples: the early tests would count the pixels it kills");
    }
    timer.phase(PhaseValidate);
    timing.Mark("validate");
    inputs.maxIndex = draw.indexed ? 0u : draw.firstVertex + draw.indexCount - 1u;
    const bool scratchInputs = RuntimeSwitches::Set(RuntimeSwitch::InputScratch);
    if (scratchInputs && recipe == nullptr) takeSpareInputs(inputs);
    inputs.minIndex = draw.indexed ? 0u : draw.firstVertex;
    if (state.quadList || StripPatches(state, shaders)) {
        Require(!draw.indirect, state.quadList ? "indirect quad-list draw is unsupported" : "an indirect triangle strip through emulated barycentric patches is unsupported");
        Require(state.quadList || !draw.indexed || !state.primitiveRestart, "an indexed triangle strip with primitive restart through emulated barycentric patches is unsupported");
        const std::uint32_t primitives = state.quadList ? draw.indexCount / 4u : draw.indexCount >= 3u ? draw.indexCount - 2u : 0u;
        if (primitives == 0u || draw.instanceCount == 0u) {
            inputs.nothing = true;
            return inputs;
        }
        const std::uint32_t corners = state.quadList ? 6u : 3u;
        Require(primitives <= std::numeric_limits<std::uint32_t>::max() / corners, "expanded index count overflow");
        const std::uint32_t outputIndexCount = primitives * corners;
        const std::size_t expandedIndexBytes = static_cast<std::size_t>(outputIndexCount) * sizeof(std::uint32_t);
        auto indexCopy = std::make_shared<Buffer>(context, expandedIndexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        auto* outIndices = reinterpret_cast<std::uint32_t*>(indexCopy->Bytes().data());
        std::uint32_t maxIdx = 0;
        const bool lastProvoking = state.provokingVertexMode == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT;
        const auto expand = [&](const auto& source) {
            for (std::uint32_t primitive = 0; primitive < primitives; ++primitive) {
                auto* out = outIndices + static_cast<std::size_t>(primitive) * corners;
                if (state.quadList) {
                    const auto i0 = source(primitive * 4u + 0u);
                    const auto i1 = source(primitive * 4u + 1u);
                    const auto i2 = source(primitive * 4u + 2u);
                    const auto i3 = source(primitive * 4u + 3u);
                    if (lastProvoking) {
                        out[0] = i0;
                        out[1] = i1;
                        out[2] = i3;
                        out[3] = i1;
                        out[4] = i2;
                        out[5] = i3;
                    } else {
                        out[0] = i0;
                        out[1] = i1;
                        out[2] = i2;
                        out[3] = i0;
                        out[4] = i2;
                        out[5] = i3;
                    }
                } else {
                    const std::uint32_t odd = primitive & 1u;
                    if (lastProvoking) {
                        out[0] = source(primitive + odd);
                        out[1] = source(primitive + 1u - odd);
                        out[2] = source(primitive + 2u);
                    } else {
                        out[0] = source(primitive);
                        out[1] = source(primitive + 1u + odd);
                        out[2] = source(primitive + 2u - odd);
                    }
                }
                for (std::uint32_t corner = 0; corner < corners; ++corner) maxIdx = std::max(maxIdx, out[corner]);
            }
        };
        if (draw.indexed) {
            GuestMemory::FlushGpuWrites(draw.indexAddress, static_cast<std::size_t>(indexBytes));
            GuestMemory::CheckRange(reinterpret_cast<const void*>(draw.indexAddress), static_cast<std::size_t>(indexBytes), draw.indexSize);
            const auto readIndex = [&](std::uint32_t idx) -> std::uint32_t {
                if (draw.indexSize == 2) {
                    std::uint16_t v = 0;
                    GuestMemory::Read(draw.indexAddress + static_cast<std::uint64_t>(idx) * 2u, std::as_writable_bytes(std::span(&v, 1)), 2);
                    return v;
                } else {
                    std::uint32_t v = 0;
                    GuestMemory::Read(draw.indexAddress + static_cast<std::uint64_t>(idx) * 4u, std::as_writable_bytes(std::span(&v, 1)), 4);
                    return v;
                }
            };
            expand(readIndex);
            Require(draw.firstVertex <= std::numeric_limits<std::uint32_t>::max() - maxIdx, "indexed draw vertex range overflow");
        } else {
            const std::uint32_t lastVertex = state.quadList ? primitives * 4u - 1u : primitives + 1u;
            Require(draw.firstVertex <= std::numeric_limits<std::uint32_t>::max() - lastVertex, "auto draw vertex range overflow");
            expand([](std::uint32_t idx) { return idx; });
        }
        inputs.maxIndex = draw.indexed ? maxIdx : maxIdx + draw.firstVertex;
        Require(inputs.maxIndex <= context.limits.maxDrawIndexedIndexValue, "index exceeds the device's indexed draw limit");
        inputs.indexHandle = indexCopy->Handle();
        inputs.indexOffset = 0;
        inputs.expandedIndexCount = outputIndexCount;
        inputs.indices = std::move(indexCopy);
    } else if (draw.indexed && state.primitiveRestart && !context.primitiveListRestart && !state.stages.mesh && (state.topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST || state.topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST || state.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)) {
        if (indexBytes == 0) {
            inputs.nothing = true;
            return inputs;
        }
        GuestMemory::FlushGpuWrites(draw.indexAddress, static_cast<std::size_t>(indexBytes));
        auto compacted = std::make_shared<Buffer>(context, static_cast<std::size_t>(indexBytes), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        const auto bytes = compacted->Bytes().first(static_cast<std::size_t>(indexBytes));
        GuestMemory::Read(draw.indexAddress, bytes, draw.indexSize);
        const auto primitiveSize = state.topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST ? 1u : state.topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ? 2u : 3u;
        const auto count = draw.indirect ? draw.indexCount : CompactListRestart(bytes, draw.indexSize, primitiveSize);
        const auto kept = bytes.first(static_cast<std::size_t>(count) * draw.indexSize);
        const bool restartLeft = draw.indirect.has_value();
        auto bounds = DrawIndexRange(kept, draw.indexSize, restartLeft);
        constexpr std::uint32_t ZeroFetchSlack = 0x10000u;
        if (const auto zeroFrom = bounds && !DrawVertexAttributes(shaders.front()).empty() ? ZeroFetchVertex(DrawVertexAttributes(shaders.front())) : std::nullopt) {
            const auto limit = *zeroFrom > draw.firstVertex ? *zeroFrom - draw.firstVertex : 0u;
            if (bounds->highest >= limit && bounds->highest - limit >= ZeroFetchSlack) bounds = ClampDrawIndices(kept, draw.indexSize, restartLeft, limit);
        }
        if (!bounds) {
            inputs.nothing = true;
            return inputs;
        }
        Require(bounds->highest <= context.limits.maxDrawIndexedIndexValue, "index exceeds the device's indexed draw limit");
        inputs.maxIndex = bounds->highest;
        inputs.minIndex = bounds->lowest;
        inputs.indexCount = count;
        inputs.listRestartPrimitive = primitiveSize;
        inputs.indices = std::move(compacted);
        inputs.indexHandle = inputs.indices->Handle();
        inputs.indexOffset = 0;
    } else if (draw.indexed) {
        const bool listTopology = state.topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST || state.topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST || state.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        const bool skipRestart = state.primitiveRestart && (!listTopology || context.primitiveListRestart || state.stages.mesh);
        const auto use = draw.indexSize == 2 ? (skipRestart ? Recorder::SnapshotUse::Index16Restart : Recorder::SnapshotUse::Index16) : (skipRestart ? Recorder::SnapshotUse::Index32Restart : Recorder::SnapshotUse::Index32);
        const bool restartTable = skipRestart && state.stages.mesh && (state.stages.mesh->inputPrimitive == 5 || state.stages.mesh->inputPrimitive == 6);
        const auto* import = InPlaceDrawInput(context, draw.indexAddress, static_cast<std::size_t>(indexBytes), draw.indexSize, nullptr, static_cast<std::size_t>(indexBytes), reads);
        DrawInputCopy copy;
        if (import == nullptr) copy = CopyDrawInput(context, context.recorder, draw.indexAddress, static_cast<std::size_t>(indexBytes), draw.indexSize, use, 0, !BindsIndexCopy(state));
        else if (!RuntimeSwitches::Set(RuntimeSwitch::SingleIndexCheck)) GuestMemory::CheckRange(reinterpret_cast<const void*>(draw.indexAddress), static_cast<std::size_t>(indexBytes), draw.indexSize);
        std::optional<std::uint32_t> highest;
        std::uint32_t lowest = 0;
        if (import != nullptr) {
            const auto range = InPlaceIndexRange(*context.recorder, draw.indexAddress, static_cast<std::size_t>(indexBytes), draw.indexSize, skipRestart);
            highest = range.highest;
            lowest = range.lowest;
        } else if (!copy.reused) {
            if (const auto bounds = DrawIndexRange(copy.buffer->Bytes().first(static_cast<std::size_t>(indexBytes)), draw.indexSize, skipRestart)) {
                highest = bounds->highest;
                lowest = bounds->lowest;
            }
            KeepDrawInput(context.recorder, draw.indexAddress, copy, use, skipRestart ? (highest ? *highest + 1u : 0u) : highest.value_or(0u), lowest);
        } else {
            lowest = copy.lowest;
            if (!skipRestart) highest = copy.derived;
            else if (copy.derived != 0) highest = copy.derived - 1u;
        }
        if (!highest) {
            inputs.nothing = true;
            return inputs;
        }
        constexpr std::uint32_t ZeroFetchSlack = 0x10000u;
        if (const auto zeroFrom = state.stages.mesh || DrawVertexAttributes(shaders.front()).empty() ? std::nullopt : ZeroFetchVertex(DrawVertexAttributes(shaders.front()))) {
            const auto limit = *zeroFrom > draw.firstVertex ? *zeroFrom - draw.firstVertex : 0u;
            if (*highest >= limit && *highest - limit >= ZeroFetchSlack) {
                GuestMemory::FlushGpuWrites(draw.indexAddress, static_cast<std::size_t>(indexBytes));
                auto clamped = std::make_shared<Buffer>(context, static_cast<std::size_t>(indexBytes), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
                const auto bytes = clamped->Bytes().first(static_cast<std::size_t>(indexBytes));
                GuestMemory::Read(draw.indexAddress, bytes, draw.indexSize);
                const auto clampedBounds = ClampDrawIndices(bytes, draw.indexSize, skipRestart, limit);
                import = nullptr;
                copy = DrawInputCopy{};
                copy.buffer = std::move(clamped);
                if (!clampedBounds) {
                    inputs.nothing = true;
                    return inputs;
                }
                highest = clampedBounds->highest;
                lowest = clampedBounds->lowest;
            }
        }
        Require(*highest <= context.limits.maxDrawIndexedIndexValue, "index exceeds the device's indexed draw limit");
        inputs.maxIndex = *highest;
        inputs.minIndex = lowest;
        if (restartTable) {
            const auto indexSpan = import != nullptr ? std::span<const std::byte>(reinterpret_cast<const std::byte*>(draw.indexAddress), static_cast<std::size_t>(indexBytes)) : copy.buffer->Bytes().first(static_cast<std::size_t>(indexBytes));
            inputs.restartTable = MeshRestartTable(indexSpan, draw.indexSize);
        }
        if (import != nullptr) {
            inputs.indexHandle = import->buffer;
            inputs.indexOffset = HostImportOffset(*import, draw.indexAddress);
            if (scratchInputs) inputs.inPlaceReads.reserve(1u + DrawVertexAttributes(shaders.front()).size());
            inputs.inPlaceReads.emplace_back(draw.indexAddress, draw.indexAddress + indexBytes);
        } else {
            inputs.indices = std::move(copy.buffer);
            inputs.indexHandle = inputs.indices->Handle();
        }
    }
    APS5_LOG_CHARS_OUT_DEBUG("Index validation OK");
    timing.Mark("index_upload");
    const auto& attributes = DrawVertexAttributes(shaders.front());
    inputs.vertexOffsets.assign(attributes.size(), 0);
    inputs.vertexHandles.reserve(attributes.size());
    // An indexed draw's vertex offset moves every fetch: the copy must reach the last one.
    if (draw.indexed) {
        Require(draw.firstVertex <= std::numeric_limits<std::uint32_t>::max() - inputs.maxIndex, "indexed draw vertex range overflow");
        inputs.maxIndex += draw.firstVertex;
        inputs.minIndex += draw.firstVertex;
    }
    auto* shared = scratchInputs ? &HostThreadLocal<DrawInputScratch, DrawInputScratch>() : nullptr;
    const bool reuse = shared != nullptr && !shared->busy;
    std::optional<VertexPlanLease> planLease;
    if (!reuse) planLease.emplace();
    auto* const lease = reuse ? nullptr : &**planLease;
    const struct ScratchHold {
        DrawInputScratch* scratch;
        ~ScratchHold() {
            if (scratch == nullptr) return;
            scratch->zeroed.clear();
            scratch->busy = false;
        }
    } scratchHold{reuse ? shared : nullptr};
    if (reuse) shared->busy = true;
    auto& fetches = reuse ? shared->fetches : lease->fetches;
    auto& fetchValid = reuse ? shared->fetchValid : lease->fetchValid;
    fetches.clear();
    fetchValid.clear();
    fetches.reserve(attributes.size());
    fetchValid.reserve(attributes.size());
    auto& fetchOf = reuse ? shared->fetchOf : lease->fetchOf;
    fetchOf.assign(attributes.size(), 0);
    auto& zeroed = reuse ? shared->zeroed : lease->zeroed;
    zeroed.clear();
    zeroed.resize(attributes.size());
    const auto& layout = inputs.VertexLayout();
    for (std::size_t i = 0; i < attributes.size(); ++i) {
        const auto& attribute = attributes[i];
        const auto described = std::find_if(layout.attributes.begin(), layout.attributes.end(), [&](const auto& entry) { return entry.location == attribute.location; });
        Require(described != layout.attributes.end(), "vertex attribute missing from its input layout");
        const auto narrow = DecodeVertexFormat(attribute);
        const std::size_t widened = described->format == narrow.format ? 0u : VertexReadBytes(attribute, described->format) - narrow.bytes;
        if (NullVertexDescriptor(attribute) || VertexFetchOutOfRange(attribute)) {
            zeroed[i] = zeroVertexBuffer(context, narrow.bytes + widened);
            continue;
        }
        // An indirect draw's counts are unknown here: the descriptor's whole range is copied.
        const auto bytes = (args != nullptr ? VertexBufferExtent(attribute) : VertexBufferReadSize(attribute, inputs.maxIndex, draw.instanceCount, draw.firstInstance)) + widened;
        const auto& fields = attribute.resource.fields;
        const auto address = fields[0] | (static_cast<std::uint64_t>(fields[1] & 0xffffu) << 32u);
        const auto firstByte = args == nullptr ? VertexBufferReadOffset(attribute, draw.firstVertex, draw.firstInstance) : 0;
        Require(!state.hasColorTarget || address + bytes <= state.color.MemoryAddress() || state.color.MemoryAddress() + state.color.bytes <= address + firstByte, "vertex buffer aliases the render target");
        const auto stride = (fields[1] >> 16u) & 0x3fffu;
        fetchOf[i] = fetches.size();
        fetches.push_back({address, address + bytes, stride, attribute.fetchIndex, DecodeVertexFormat(attribute).alignment});
        const auto recordBytes = static_cast<std::uint64_t>(fields[2]) * stride;
        fetchValid.push_back(args == nullptr && stride != 0 ? static_cast<std::size_t>(std::min(recordBytes, static_cast<std::uint64_t>(bytes))) : bytes);
    }
    std::vector<std::size_t> ownSolo;
    auto& soloFetches = reuse ? shared->soloFetches : ownSolo;
    SoloZeroPaddedFetchIndicesInto(fetches, fetchValid, soloFetches);
    auto& isSolo = reuse ? shared->isSolo : lease->isSolo;
    isSolo.assign(fetches.size(), false);
    for (const auto f : soloFetches) isSolo[f] = true;
    auto& plannedFetches = reuse ? shared->plannedFetches : lease->plannedFetches;
    plannedFetches.clear();
    plannedFetches.reserve(fetches.size());
    auto& plannedOf = reuse ? shared->plannedOf : lease->plannedOf;
    plannedOf.assign(fetches.size(), 0);
    for (std::size_t f = 0; f < fetches.size(); ++f) {
        if (isSolo[f] != 0) continue;
        plannedOf[f] = plannedFetches.size();
        plannedFetches.push_back(fetches[f]);
    }
    VertexCopyPlan ownPlan;
    if (reuse) PlanVertexCopiesInto(plannedFetches, shared->plan, shared->order);
    else ownPlan = PlanVertexCopies(plannedFetches);
    const auto& plan = reuse ? shared->plan : ownPlan;
    const auto rangeFrom = args == nullptr && !state.stages.mesh ? VertexCopyStarts(plan, plannedFetches, inputs.minIndex, draw.firstInstance) : std::vector<std::size_t>(plan.copies.size(), 0);
    auto& rangeHandles = reuse ? shared->rangeHandles : lease->rangeHandles;
    auto& rangeOffsets = reuse ? shared->rangeOffsets : lease->rangeOffsets;
    rangeHandles.assign(plan.copies.size(), VK_NULL_HANDLE);
    rangeOffsets.assign(plan.copies.size(), 0);
    for (std::size_t range = 0; range < plan.copies.size(); ++range) {
        const auto [begin, end] = plan.copies[range];
        const auto bytes = static_cast<std::size_t>(end - begin);
        GuestMemory::CheckRange(reinterpret_cast<const void*>(begin), bytes, 1);
        RequireNoMultisampleTarget(begin, bytes, "a vertex buffer read");
        if (const auto* import = InPlaceDrawInput(context, begin, bytes, plan.alignments[range], &inputs.vertexInputBehindGpuWrite, bytes - std::min(rangeFrom[range], bytes), reads)) {
            rangeHandles[range] = import->buffer;
            rangeOffsets[range] = HostImportOffset(*import, begin);
            if (scratchInputs) inputs.inPlaceReads.reserve(inputs.inPlaceReads.size() + plan.copies.size() - range);
            inputs.inPlaceReads.emplace_back(begin, end);
            continue;
        }
        auto copy = CopyDrawInput(context, context.recorder, begin, bytes, 1, Recorder::SnapshotUse::Vertex, rangeFrom[range]);
        KeepDrawInput(context.recorder, begin, copy, Recorder::SnapshotUse::Vertex, 0);
        rangeHandles[range] = copy.buffer->Handle();
        rangeOffsets[range] = 0;
        inputs.vertexBuffers.push_back(std::move(copy.buffer));
    }
    std::vector<VkBuffer> ownSoloHandles;
    auto& soloHandles = reuse ? shared->soloHandles : ownSoloHandles;
    soloHandles.assign(fetches.size(), VK_NULL_HANDLE);
    for (const auto f : soloFetches) {
        const auto begin = fetches[f].begin;
        const auto bytes = static_cast<std::size_t>(fetches[f].end - begin);
        auto copy = CopyZeroPaddedDrawInput(context, begin, bytes, fetchValid[f], fetches[f].stride);
        soloHandles[f] = copy.buffer->Handle();
        inputs.vertexBuffers.push_back(std::move(copy.buffer));
    }
    for (std::size_t i = 0; i < attributes.size(); ++i) {
        if (zeroed[i] != nullptr) {
            inputs.vertexHandles.push_back(zeroed[i]->Handle());
            inputs.vertexOffsets[i] = 0;
            inputs.vertexBuffers.push_back(std::move(zeroed[i]));
            continue;
        }
        const auto f = fetchOf[i];
        if (isSolo[f] != 0) {
            inputs.vertexHandles.push_back(soloHandles[f]);
            inputs.vertexOffsets[i] = 0;
            continue;
        }
        const auto j = plannedOf[f];
        inputs.vertexHandles.push_back(rangeHandles[plan.copyOf[j]]);
        inputs.vertexOffsets[i] = rangeOffsets[plan.copyOf[j]] + plan.offsets[j];
    }
    timer.phase(PhaseVertex);
    timing.Mark("vertex_upload");
    return inputs;
}

struct PreparedInputScratch {
    std::vector<VertexFetch> fetches;
    std::vector<std::size_t> fetchValid;
    std::vector<std::size_t> solo;
    VertexCopyPlan plan;
    std::vector<std::size_t> order;
    std::vector<ExistingHostImport> found;
};

bool sameVertexLayout(const VertexInputLayout& a, const VertexInputLayout& b) {
    const auto sameBinding = [](const VkVertexInputBindingDescription& x, const VkVertexInputBindingDescription& y) { return x.binding == y.binding && x.stride == y.stride && x.inputRate == y.inputRate; };
    const auto sameAttribute = [](const VkVertexInputAttributeDescription& x, const VkVertexInputAttributeDescription& y) { return x.location == y.location && x.binding == y.binding && x.format == y.format && x.offset == y.offset; };
    return std::ranges::equal(a.bindings, b.bindings, sameBinding) && std::ranges::equal(a.attributes, b.attributes, sameAttribute);
}

bool inputRangeHazard(const Recorder& recorder, std::uint64_t begin, std::size_t bytes) {
    return recorder.QueuedStoreOverlaps(begin, bytes) || (!PendingZones::Clear(begin, bytes) && (recorder.PendingWriteOverlaps(begin, bytes) || PendingStorageOverlaps(begin, bytes, nullptr))) || recorder.PendingLabelIn(begin, bytes);
}

// Whether prepared inputs may be taken where draw inputs are not bound in place (APS5_MIRRORED_INPUTS).
bool preparedInputsUsable(const Context& context, const PreparedInputs& prepared) {
    return context.drawInputsInPlace || (prepared.mirrored && RuntimeSwitches::Set(RuntimeSwitch::MirroredInputs));
}

// The commit's test of a range bound from a published copy. A copy is not read behind a GPU write and
// no order of commands makes it current, so nothing here is skipped by a trusted or audited committer:
// the guards the copy path requires, no pending GPU work over the range (the copy path would wait for
// it and copy again), then the proof that the guest bytes are still the copied ones.
bool mirroredInputHolds(const Context& context, const PreparedInputs::Range& range, bool index) {
    auto* recorder = context.recorder;
    const auto bytes = static_cast<std::size_t>(range.end - range.begin);
    if (recorder == nullptr || range.source == nullptr || bytes == 0 || !RuntimeSwitches::Set(RuntimeSwitch::MirroredInputs)) return false;
    RequireNoMultisampleTarget(range.begin, bytes, index ? "an index buffer read" : "a vertex buffer read");
    const auto overlaps = [&](const auto& writer) { return writer->WritesOverlap(range.begin, bytes); };
    const auto drawWriters = DrawCopiedWriters();
    // The exact tests, not the zone table the in-place routes ask first.
    if (recorder->PendingWriteOverlaps(range.begin, bytes) || PendingStorageOverlaps(range.begin, bytes, nullptr) || recorder->PendingLabelIn(range.begin, bytes) || recorder->QueuedStoreOverlaps(range.begin, bytes) || (context.copiedWriters != nullptr && std::any_of(context.copiedWriters->begin(), context.copiedWriters->end(), overlaps)) || std::any_of(drawWriters->begin(), drawWriters->end(), overlaps)) {
        PerformanceTrace::CountOwnedDraw("OwnedDraw.mirror_hazard");
        return false;
    }
    return recorder->PublishedDrawInputHolds(range.begin, bytes, range.ticket);
}

bool inputRangeAnswered(const Recorder& recorder, std::uint64_t begin, std::size_t bytes, WriteValidation::Test test) {
    const auto answer = WriteValidation::CommitterClear(recorder, begin, bytes, WriteValidation::Fact::Inputs, test);
    if (answer == WriteValidation::Answer::Today) return false;
    if (answer == WriteValidation::Answer::Proved || !WriteValidation::VerifyDue(test)) return true;
    const bool hazard = inputRangeHazard(recorder, begin, bytes);
    WriteValidation::NoteVerify(test, hazard, "an input range has a queued store, a pending write, a pending storage image or a pending label");
    return !hazard;
}

bool inputGuardsAnswered(const Recorder* recorder, std::uint64_t address, std::size_t bytes, WriteValidation::Test test) {
    if (recorder == nullptr || WriteValidation::Session() == 0) return false;
    const auto answer = WriteValidation::CommitterClear(*recorder, address, bytes, WriteValidation::Fact::Guards, test);
    if (answer == WriteValidation::Answer::Today) return false;
    if (answer == WriteValidation::Answer::Proved || !WriteValidation::VerifyDue(test)) return true;
    const bool idle = DepthReadGuardIdle_nid_no_patch(address, bytes) && !MayOverlapMultisampleGuards(address, bytes);
    WriteValidation::NoteVerify(test, !idle, "an input range overlaps a depth or multisample guard, or an overwrite proof is pending");
    return idle;
}

void requireInputGuards(CommitterTrustCheck* trust, std::uint64_t address, std::size_t bytes, const char* access) {
    if (trust != nullptr && trust->Audits() && (!DepthReadGuardIdle_nid_no_patch(address, bytes) || MayOverlapMultisampleGuards(address, bytes))) trust->Fail();
    RequireNoMultisampleTarget(address, bytes, access);
}

CommitterTrustCheck committerTrustCheck(const ShaderResources& resources, CommitterTrustFamily family, CommitterTrustSite site) {
    if (!RuntimeSwitches::Set(RuntimeSwitch::TrustAudit) || RuntimeSwitches::Set(RuntimeSwitch::XpTrustCommitter)) return {};
    return resources.TrustCheck(family, site);
}

DrawRecordInputs takePreparedInputs(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, const DrawRecipe& recipe, PreparedInputs& prepared, DrawInputs& inputs, DrawTimer& timer, PerformanceTimer& timing, bool atFloor = false, bool validated = false, CommitterTrustCheck* trust = nullptr) {
    const std::size_t indexRanges = draw.indexed ? 1u : 0u;
    if (prepared.recipe.get() != &recipe || prepared.device != context.device || prepared.state != &state || prepared.draw != &draw || prepared.shaders != shaders.data() || prepared.stages != shaders.size() || prepared.ranges.size() < indexRanges || context.recorder == nullptr || !preparedInputsUsable(context, prepared)) return DrawRecordInputs::Shape;
    const bool mapped = (prepared.forget & 1u) == 0 && GuestMemory::ForgetSerial() == prepared.forget;
    const auto indexBytes = static_cast<std::size_t>(prepared.indexBytes);
    const bool trusted = !atFloor && RuntimeSwitches::Set(RuntimeSwitch::XpTrustCommitter);
    if (trusted) NoteCommitterTrusted();
    const bool guardsSkipped = trusted || (!atFloor && trust != nullptr && trust->Skip());
    if (draw.indexed && !atFloor) {
        if (!mapped) GuestMemory::CheckRange(reinterpret_cast<const void*>(draw.indexAddress), indexBytes, draw.indexSize);
        if (!guardsSkipped && !inputGuardsAnswered(context.recorder, draw.indexAddress, indexBytes, WriteValidation::Test::LeaderGuards)) requireInputGuards(trust, draw.indexAddress, indexBytes, "an index buffer read");
    }
    if (!atFloor && !validated) {
        requireUncountedKills(state);
        ValidateViewport(context, state.viewport);
        ValidateDepthBounds(context, state);
        ValidateSampleMask(context, state, shaders);
    }
    timer.phase(PhaseValidate);
    timing.Mark("validate");
    bool rechecked = false;
    const auto served = [&](const PreparedInputs::Range& range, bool* behind) {
        if (range.source != nullptr) return mirroredInputHolds(context, range, behind == nullptr) ? DrawRecordInputs::Used : DrawRecordInputs::Refused;
        if (atFloor) return DrawRecordInputs::Used;
        KnownImport known{range.import, prepared.imports, prepared.mapping};
        const auto* import = inPlaceDrawInput(context, range.begin, static_cast<std::size_t>(range.end - range.begin), range.alignment, behind, &known, range.small ? std::size_t{0} : std::numeric_limits<std::size_t>::max(), range.small ? std::uint64_t{0} : std::numeric_limits<std::uint64_t>::max());
        rechecked = rechecked || known.looked;
        if (import == nullptr) return DrawRecordInputs::Refused;
        return import->buffer == range.handle && HostImportOffset(*import, range.begin) == range.offset ? DrawRecordInputs::Used : DrawRecordInputs::ImportChanged;
    };
    if (draw.indexed) {
        if (const auto use = served(prepared.ranges.front(), nullptr); use != DrawRecordInputs::Used) return use;
        if (!mapped && !RuntimeSwitches::Set(RuntimeSwitch::SingleIndexCheck)) GuestMemory::CheckRange(reinterpret_cast<const void*>(draw.indexAddress), indexBytes, draw.indexSize);
    }
    timing.Mark("index_upload");
    bool behind = false;
    for (std::size_t at = indexRanges; at < prepared.ranges.size(); ++at) {
        const auto& range = prepared.ranges[at];
        const auto bytes = static_cast<std::size_t>(range.end - range.begin);
        if (!mapped) GuestMemory::CheckRange(reinterpret_cast<const void*>(range.begin), bytes, 1);
        if (!atFloor && !guardsSkipped && at + prepared.indirectRanges < prepared.ranges.size() && !inputGuardsAnswered(context.recorder, range.begin, bytes, WriteValidation::Test::LeaderGuards)) requireInputGuards(trust, range.begin, bytes, "a vertex buffer read");
        if (const auto use = served(range, &behind); use != DrawRecordInputs::Used) return use;
    }
    inputs.indexBytes = prepared.indexBytes;
    if (draw.indexed) {
        inputs.indexHandle = prepared.ranges.front().handle;
        inputs.indexOffset = prepared.ranges.front().offset;
    }
    inputs.maxIndex = prepared.maxIndex;
    inputs.vertexInput = std::move(prepared.vertexInput);
    inputs.vertexHandles = std::move(prepared.vertexHandles);
    inputs.vertexOffsets = std::move(prepared.vertexOffsets);
    inputs.inPlaceReads = std::move(prepared.inPlaceReads);
    for (std::size_t at = 0; prepared.mirrored && at < prepared.ranges.size(); ++at) {
        const auto& source = prepared.ranges[at].source;
        if (source == nullptr) continue;
        context.recorder->Keep(source);
        if (at < indexRanges) inputs.indices = source;
        else inputs.vertexBuffers.push_back(source);
    }
    inputs.vertexInputBehindGpuWrite = behind;
    inputs.fragmentOutputs = recipe.fragmentOutputs;
    inputs.shaderStages = recipe.shaderStages;
    timer.phase(PhaseVertex);
    timing.Mark("vertex_upload");
    return rechecked ? DrawRecordInputs::Rechecked : DrawRecordInputs::Used;
}

std::string describeInputsDifference(const DrawInputs& mine, const DrawInputs& today) {
    char text[320];
    const auto wide = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    if (mine.nothing != today.nothing) {
        std::snprintf(text, sizeof(text), "nothing to draw: prepared %d, today's %d", mine.nothing ? 1 : 0, today.nothing ? 1 : 0);
        return text;
    }
    if (mine.indexBytes != today.indexBytes) {
        std::snprintf(text, sizeof(text), "index bytes: prepared %llu, today's %llu", wide(mine.indexBytes), wide(today.indexBytes));
        return text;
    }
    if ((mine.indices != nullptr) != (today.indices != nullptr)) {
        std::snprintf(text, sizeof(text), "index buffer: prepared %s, today's %s", mine.indices != nullptr ? "copied" : "in place", today.indices != nullptr ? "copied" : "in place");
        return text;
    }
    if (mine.indexHandle != today.indexHandle || mine.indexOffset != today.indexOffset) {
        std::snprintf(text, sizeof(text), "index buffer: prepared %s buffer at offset %llu, today's at %llu", mine.indexHandle == today.indexHandle ? "the same" : "another", wide(mine.indexOffset), wide(today.indexOffset));
        return text;
    }
    if (mine.maxIndex != today.maxIndex) {
        std::snprintf(text, sizeof(text), "max index: prepared %u, today's %u", mine.maxIndex, today.maxIndex);
        return text;
    }
    if (mine.indexCount != today.indexCount) {
        std::snprintf(text, sizeof(text), "index count: prepared %lld, today's %lld", mine.indexCount ? static_cast<long long>(*mine.indexCount) : -1ll, today.indexCount ? static_cast<long long>(*today.indexCount) : -1ll);
        return text;
    }
    if (mine.listRestartPrimitive != today.listRestartPrimitive || mine.expandedIndexCount != today.expandedIndexCount || mine.restartTable != today.restartTable) {
        std::snprintf(text, sizeof(text), "rewritten indices: prepared restart primitive %u, %u expanded, %zu restart starts; today's %u, %u, %zu", mine.listRestartPrimitive, mine.expandedIndexCount, mine.restartTable.size(), today.listRestartPrimitive, today.expandedIndexCount, today.restartTable.size());
        return text;
    }
    if ((mine.retainedVertexInput != nullptr) != (today.retainedVertexInput != nullptr) || !sameVertexLayout(mine.VertexLayout(), today.VertexLayout())) {
        std::snprintf(text, sizeof(text), "vertex layout: prepared %zu bindings and %zu attributes, today's %zu and %zu", mine.VertexLayout().bindings.size(), mine.VertexLayout().attributes.size(), today.VertexLayout().bindings.size(), today.VertexLayout().attributes.size());
        return text;
    }
    if (mine.vertexBuffers.size() != today.vertexBuffers.size()) {
        std::snprintf(text, sizeof(text), "vertex copies: prepared %zu, today's %zu", mine.vertexBuffers.size(), today.vertexBuffers.size());
        return text;
    }
    if (mine.vertexHandles.size() != today.vertexHandles.size() || mine.vertexOffsets.size() != today.vertexOffsets.size()) {
        std::snprintf(text, sizeof(text), "vertex bindings: prepared %zu buffers and %zu offsets, today's %zu and %zu", mine.vertexHandles.size(), mine.vertexOffsets.size(), today.vertexHandles.size(), today.vertexOffsets.size());
        return text;
    }
    for (std::size_t index = 0; index < mine.vertexHandles.size(); ++index) {
        if (mine.vertexHandles[index] == today.vertexHandles[index] && mine.vertexOffsets[index] == today.vertexOffsets[index]) continue;
        std::snprintf(text, sizeof(text), "vertex binding %zu: prepared %s buffer at offset %llu, today's at %llu", index, mine.vertexHandles[index] == today.vertexHandles[index] ? "the same" : "another", wide(mine.vertexOffsets[index]), wide(today.vertexOffsets[index]));
        return text;
    }
    if (mine.inPlaceReads != today.inPlaceReads) {
        std::size_t index = 0;
        while (index < mine.inPlaceReads.size() && index < today.inPlaceReads.size() && mine.inPlaceReads[index] == today.inPlaceReads[index]) ++index;
        const auto range = [&](const DrawInputs& inputs) { return index < inputs.inPlaceReads.size() ? inputs.inPlaceReads[index] : std::pair<std::uint64_t, std::uint64_t>{0, 0}; };
        std::snprintf(text, sizeof(text), "in-place reads: prepared %zu, today's %zu; at %zu prepared 0x%llx-0x%llx, today's 0x%llx-0x%llx", mine.inPlaceReads.size(), today.inPlaceReads.size(), index, wide(range(mine).first), wide(range(mine).second), wide(range(today).first), wide(range(today).second));
        return text;
    }
    if (mine.vertexInputBehindGpuWrite != today.vertexInputBehindGpuWrite) {
        std::snprintf(text, sizeof(text), "vertex input behind a GPU write: prepared %d, today's %d", mine.vertexInputBehindGpuWrite ? 1 : 0, today.vertexInputBehindGpuWrite ? 1 : 0);
        return text;
    }
    if (mine.fragmentOutputs != today.fragmentOutputs || mine.shaderStages != today.shaderStages) {
        std::snprintf(text, sizeof(text), "fragment outputs and stages: prepared 0x%x and 0x%x, today's 0x%x and 0x%x", mine.fragmentOutputs, static_cast<unsigned>(mine.shaderStages), today.fragmentOutputs, static_cast<unsigned>(today.shaderStages));
        return text;
    }
    if (mine.meshGroups != today.meshGroups || mine.meshLayers != today.meshLayers || mine.meshLayerStride != today.meshLayerStride) {
        std::snprintf(text, sizeof(text), "mesh grid: prepared %u x %u by %u, today's %u x %u by %u", mine.meshGroups, mine.meshLayers, mine.meshLayerStride, today.meshGroups, today.meshLayers, today.meshLayerStride);
        return text;
    }
    return {};
}

DrawRecordInputs verifyPreparedInputs(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, const DrawRecipe& recipe, PreparedInputs& prepared, const DrawInputs& today, DrawTimer& timer, PerformanceTimer& timing, std::string& difference, DrawRecordInputs* taken = nullptr) {
    DrawInputs mine;
    const auto use = takePreparedInputs(context, state, draw, shaders, recipe, prepared, mine, timer, timing);
    if (taken != nullptr) *taken = use;
    if (use != DrawRecordInputs::Used && use != DrawRecordInputs::Rechecked) return use;
    if (draw.indirect && prepared.indirectRanges <= mine.inPlaceReads.size()) {
        mine.inPlaceReads.resize(mine.inPlaceReads.size() - prepared.indirectRanges);
        mine.maxIndex = today.maxIndex;
    }
    difference = describeInputsDifference(mine, today);
    if (difference.empty() && draw.indirect) {
        for (std::size_t at = prepared.ranges.size() - std::min(prepared.indirectRanges, prepared.ranges.size()); at < prepared.ranges.size() && difference.empty(); ++at) {
            const auto& range = prepared.ranges[at];
            const auto* import = HostImportFor(context, range.begin, static_cast<std::size_t>(range.end - range.begin));
            if (import != nullptr && import->buffer == range.handle && HostImportOffset(*import, range.begin) == range.offset) continue;
            char text[200];
            std::snprintf(text, sizeof(text), "indirect %s at 0x%llx: prepared %s buffer at offset %llu, today's %s", at + 1u == prepared.ranges.size() && prepared.indirectRanges == 2u ? "count" : "records", static_cast<unsigned long long>(range.begin), import != nullptr && import->buffer == range.handle ? "the same" : "another", static_cast<unsigned long long>(range.offset), import != nullptr ? "binds another offset or buffer" : "finds no import");
            difference = text;
        }
        if (difference.empty() && prepared.indirectRanges == 0) difference = "indirect records: the prepared inputs name none";
    }
    return difference.empty() ? DrawRecordInputs::Checked : DrawRecordInputs::Differs;
}

void noteInputsDifference(const std::string& difference) {
    static std::atomic<std::uint64_t> printed{0};
    if (printed.fetch_add(1, std::memory_order_relaxed) < 20) std::fprintf(stderr, "[direct-inputs] verify: %s\n", difference.c_str());
}

struct InputsMemoEntry {
    std::shared_ptr<const DrawRecipe> recipe;
    std::vector<ShaderRecompiler::VertexAttribute> attributes;
    std::uint32_t firstVertex = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t firstInstance = 0;
    std::uint32_t maxIndex = 0;
    std::uint32_t indexSize = 0;
    bool indexed = false;
    bool hasColor = false;
    std::uint64_t colorBegin = 0;
    std::uint64_t colorBytes = 0;
    std::uint64_t forget = 0;
    std::uint64_t imports = 0;
    std::uint64_t mapping = 0;
    std::vector<PreparedInputs::Range> ranges;
    std::vector<VkBuffer> handles;
    std::vector<VkDeviceSize> offsets;
    bool indexKnown = false;
    std::uint64_t indexBegin = 0;
    std::uint64_t indexEnd = 0;
    ExistingHostImport indexFound{};
};

struct InputsMemo {
    static constexpr std::size_t Slots = 512;
    std::vector<InputsMemoEntry> slots;
    VkDevice device = VK_NULL_HANDLE;
    std::uint64_t ordinal = 0;
    InputsMemoNote note;
};

struct InputsMemoStorage {};

std::atomic<bool> inputsMemoUsed{false};

std::uint64_t inputsMemoVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_DIRECT_INPUTS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

bool sameMemoAttribute(const ShaderRecompiler::VertexAttribute& a, const ShaderRecompiler::VertexAttribute& b) {
    return a.location == b.location && a.components == b.components && a.fetchIndex == b.fetchIndex && a.formatComponents == b.formatComponents && a.resource.fields == b.resource.fields;
}

std::size_t inputsMemoSlot(const DrawRecipe* recipe, std::span<const ShaderRecompiler::VertexAttribute> attributes, const Pm4::DrawParameters& draw, std::uint32_t maxIndex) {
    std::uint64_t hash = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(recipe)) * 0x9e3779b97f4a7c15ull;
    const auto mix = [&](std::uint64_t value) { hash = (hash ^ value) * 0x100000001b3ull; };
    mix((static_cast<std::uint64_t>(draw.firstVertex) << 32u) | maxIndex);
    mix((static_cast<std::uint64_t>(draw.instanceCount) << 32u) | draw.firstInstance);
    for (const auto& attribute : attributes) {
        mix((static_cast<std::uint64_t>(attribute.resource.fields[0]) << 32u) | attribute.resource.fields[1]);
        mix((static_cast<std::uint64_t>(attribute.resource.fields[2]) << 32u) | attribute.resource.fields[3]);
        mix(attribute.location);
    }
    hash ^= hash >> 29u;
    return static_cast<std::size_t>(hash % InputsMemo::Slots);
}

bool sameInputsKey(const InputsMemoEntry& entry, std::span<const ShaderRecompiler::VertexAttribute> attributes, const Pm4::DrawParameters& draw, std::uint32_t maxIndex, const State& state) {
    if (entry.firstVertex != draw.firstVertex || entry.instanceCount != draw.instanceCount || entry.firstInstance != draw.firstInstance || entry.maxIndex != maxIndex || entry.indexed != draw.indexed || entry.indexSize != draw.indexSize) return false;
    if (entry.hasColor != state.hasColorTarget) return false;
    if (state.hasColorTarget && (entry.colorBegin != static_cast<std::uint64_t>(state.color.MemoryAddress()) || entry.colorBytes != static_cast<std::uint64_t>(state.color.bytes))) return false;
    return std::ranges::equal(entry.attributes, attributes, sameMemoAttribute);
}

const char* preparedInputsDifference(const PreparedInputs& mine, const PreparedInputs& today) {
    if (mine.refusal != today.refusal) return "the refusal";
    if (mine.indexBytes != today.indexBytes || mine.maxIndex != today.maxIndex) return "the index range";
    if (mine.ranges.size() != today.ranges.size()) return "the range count";
    for (std::size_t at = 0; at < mine.ranges.size(); ++at) {
        const auto& a = mine.ranges[at];
        const auto& b = today.ranges[at];
        if (a.begin != b.begin || a.end != b.end || a.alignment != b.alignment || a.import != b.import || a.handle != b.handle || a.offset != b.offset) return "a range";
    }
    if (mine.vertexHandles != today.vertexHandles || mine.vertexOffsets != today.vertexOffsets) return "the vertex bindings";
    if (mine.inPlaceReads != today.inPlaceReads) return "the in-place reads";
    if (!sameVertexLayout(mine.vertexInput, today.vertexInput)) return "the vertex layout";
    return nullptr;
}

bool prepareMemoInputs(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::shared_ptr<const DrawRecipe> recipe, PreparedInputs& prepared, InputsMemo* memo);

}

InputsMemoNote TakeInputsMemoNote() {
    auto& memo = HostThreadLocal<InputsMemo, InputsMemoStorage>();
    return std::exchange(memo.note, {});
}

bool PrepareDrawInputs(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::shared_ptr<const DrawRecipe> recipe, PreparedInputs& prepared) {
    InputsMemo* memo = nullptr;
    if (RuntimeSwitches::Set(RuntimeSwitch::InputsMemo)) {
        memo = &HostThreadLocal<InputsMemo, InputsMemoStorage>();
        inputsMemoUsed.store(true, std::memory_order_relaxed);
    } else if (inputsMemoUsed.load(std::memory_order_relaxed)) {
        auto& held = HostThreadLocal<InputsMemo, InputsMemoStorage>();
        if (!held.slots.empty()) held.slots.clear();
    }
    return prepareMemoInputs(context, state, draw, shaders, std::move(recipe), prepared, memo);
}

namespace {

bool prepareMemoInputs(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::shared_ptr<const DrawRecipe> recipe, PreparedInputs& prepared, InputsMemo* memo) {
    using Refusal = PreparedInputs::Refusal;
    const auto refuse = [&](Refusal why) {
        prepared.refusal = why;
        return false;
    };
    prepared.refusal = Refusal::None;
    prepared.state = &state;
    prepared.draw = &draw;
    prepared.shaders = shaders.data();
    prepared.stages = shaders.size();
    prepared.device = context.device;
    prepared.forget = GuestMemory::ForgetSerial();
    prepared.imports = HostImportSetGeneration();
    prepared.mapping = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    prepared.vertexHandles.clear();
    prepared.vertexOffsets.clear();
    prepared.inPlaceReads.clear();
    prepared.ranges.clear();
    prepared.indirectRanges = 0;
    prepared.mirrored = false;
    if (recipe == nullptr || recipe->device != context.device) return refuse(Refusal::NoRecipe);
    // APS5_MIRRORED_INPUTS: where inputs are copied, a range is bound from the copy the committer
    // published, or in place when the committer would bind it so (a small range). The inputs memo keys
    // neither the lowest index nor the publications and an indirect draw's inputs are copied whole: both
    // stay with the committer.
    const bool mirrored = !context.drawInputsInPlace && RuntimeSwitches::Set(RuntimeSwitch::MirroredInputs);
    if (context.recorder == nullptr || (!context.drawInputsInPlace && !mirrored) || context.hostImportAlignment == 0) return refuse(Refusal::NotInPlace);
    if (mirrored && draw.indirect) return refuse(Refusal::NotInPlace);
    if (mirrored) memo = nullptr;
    const auto* args = draw.indirect ? &*draw.indirect : nullptr;
    if (args != nullptr && (!RuntimeSwitches::Set(RuntimeSwitch::DirectGpuIndirect) || args->RangeBytes() == 0 || args->stride % 4u != 0 || args->arguments % 4u != 0 || (draw.indexed && !args->indexLimitKnown) || (args->countIndirect && (!context.drawIndirectCount || args->countAddress % 4u != 0)))) return refuse(Refusal::Shape);
    if (shaders.empty() || state.quadList || state.stages.mesh || state.stages.tessellation || state.stages.tessellationGeometry || state.tessellationCapture != nullptr || StripPatches(state, shaders)) return refuse(Refusal::Shape);
    Require(draw.indexed ? draw.flags == 0 : (draw.flags & ~0x20u) == 0, "draw modifiers are unsupported");
    if (draw.indexed) {
        Require(draw.indexSize == 2 || draw.indexSize == 4, "only uint16 and uint32 index buffers are supported");
    } else {
        Require(draw.indexAddress == 0 && draw.indexSize == 0, "auto draw must not reference an index buffer");
        if (args == nullptr) {
            if (draw.indexCount == 0 || draw.instanceCount == 0) return refuse(Refusal::Shape);
            Require(draw.firstVertex <= std::numeric_limits<std::uint32_t>::max() - (draw.indexCount - 1u), "auto draw vertex range overflow");
            Require(draw.firstInstance <= std::numeric_limits<std::uint32_t>::max() - (draw.instanceCount - 1u), "auto draw instance range overflow");
        }
    }
    if (args == nullptr && (draw.indexCount == 0 || draw.instanceCount == 0)) return refuse(Refusal::Shape);
    if (args != nullptr && draw.indexed && draw.indexCount == 0) return refuse(Refusal::Shape);
    const auto indexBytes = static_cast<std::uint64_t>(draw.indexCount) * draw.indexSize;
    Require(indexBytes <= std::numeric_limits<std::size_t>::max(), "index buffer size overflow");
    if (draw.indexed && (draw.indexAddress % draw.indexSize != 0 || !GuestMemory::Accessible(reinterpret_cast<const void*>(draw.indexAddress), static_cast<std::size_t>(indexBytes)))) return refuse(Refusal::Inaccessible);
    Require(!draw.indexed || !state.hasColorTarget || draw.indexAddress + indexBytes <= state.color.MemoryAddress() || state.color.MemoryAddress() + state.color.bytes <= draw.indexAddress, "index buffer aliases the render target");
    const auto& attributes = DrawVertexAttributes(shaders.front());
    const bool capacity = RuntimeSwitches::Set(RuntimeSwitch::PreparedCapacity);
    prepared.exactCapacity = capacity;
    if (capacity) prepared.ranges.reserve((draw.indexed ? 1u : 0u) + attributes.size() + 2u);
    std::uint32_t maxIndex = draw.indexed || args != nullptr ? 0u : draw.firstVertex + draw.indexCount - 1u;
    std::uint32_t minIndex = draw.firstVertex;
    Recorder::DrawInputTicket indexTicket;
    if (draw.indexed) {
        const bool listTopology = state.topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST || state.topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST || state.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        if (state.primitiveRestart && !context.primitiveListRestart && listTopology) return refuse(Refusal::Shape);
        if (args != nullptr) {
            if (draw.indexSize != 2u && context.limits.maxDrawIndexedIndexValue != std::numeric_limits<std::uint32_t>::max()) return refuse(Refusal::Shape);
            if (!attributes.empty() && ZeroFetchVertex(attributes).has_value()) return refuse(Refusal::Shape);
        } else {
            const bool skipRestart = state.primitiveRestart && (!listTopology || context.primitiveListRestart);
            const auto bounds = DrawIndexRange(std::span<const std::byte>(reinterpret_cast<const std::byte*>(draw.indexAddress), static_cast<std::size_t>(indexBytes)), draw.indexSize, skipRestart);
            const auto highest = bounds ? std::optional<std::uint32_t>(bounds->highest) : std::nullopt;
            if (!highest) return refuse(Refusal::Shape);
            constexpr std::uint32_t ZeroFetchSlack = 0x10000u;
            if (const auto zeroFrom = attributes.empty() ? std::nullopt : ZeroFetchVertex(attributes)) {
                const auto limit = *zeroFrom > draw.firstVertex ? *zeroFrom - draw.firstVertex : 0u;
                if (*highest >= limit && *highest - limit >= ZeroFetchSlack) return refuse(Refusal::Shape);
            }
            Require(*highest <= context.limits.maxDrawIndexedIndexValue, "index exceeds the device's indexed draw limit");
            Require(draw.firstVertex <= std::numeric_limits<std::uint32_t>::max() - *highest, "indexed draw vertex range overflow");
            maxIndex = *highest + draw.firstVertex;
            minIndex = bounds->lowest + draw.firstVertex;
            indexTicket.derived = skipRestart ? *highest + 1u : *highest;
            indexTicket.lowest = bounds->lowest;
            indexTicket.use = draw.indexSize == 2 ? (skipRestart ? Recorder::SnapshotUse::Index16Restart : Recorder::SnapshotUse::Index16) : (skipRestart ? Recorder::SnapshotUse::Index32Restart : Recorder::SnapshotUse::Index32);
        }
        prepared.ranges.push_back({draw.indexAddress, draw.indexAddress + indexBytes, draw.indexSize});
    }
    prepared.vertexInput = recipe->vertexInput;
    const auto& layout = prepared.vertexInput;
    InputsMemoEntry* slot = nullptr;
    if (memo != nullptr && args != nullptr) memo->note.outcome = InputsMemoOutcome::Indirect;
    if (memo != nullptr && args == nullptr) {
        if (memo->device != context.device) {
            memo->slots.clear();
            memo->device = context.device;
        }
        if (memo->slots.empty()) memo->slots.resize(InputsMemo::Slots);
        slot = &memo->slots[inputsMemoSlot(recipe.get(), attributes, draw, maxIndex)];
        const bool keyed = slot->recipe == recipe && sameInputsKey(*slot, attributes, draw, maxIndex, state);
        const bool current = (prepared.forget & 1u) == 0 && slot->forget == prepared.forget && slot->imports == prepared.imports && slot->mapping == prepared.mapping;
        memo->note.outcome = !keyed ? InputsMemoOutcome::MissKey : !current ? InputsMemoOutcome::MissSerial : InputsMemoOutcome::Hit;
        if (keyed && current) {
            for (const auto& range : slot->ranges) {
                if (!GuestMemory::Accessible(reinterpret_cast<const void*>(range.begin), static_cast<std::size_t>(range.end - range.begin))) return refuse(Refusal::Inaccessible);
            }
            if (draw.indexed) {
                auto& index = prepared.ranges.front();
                if (!slot->indexKnown || slot->indexBegin != index.begin || slot->indexEnd != index.end) {
                    const std::pair<std::uint64_t, std::uint64_t> read{index.begin, index.end};
                    ExistingHostImport found{};
                    if (!ExistingHostImportsFor(context, std::span(&read, 1), std::span(&found, 1))) return refuse(Refusal::NotInPlace);
                    slot->indexKnown = true;
                    slot->indexBegin = index.begin;
                    slot->indexEnd = index.end;
                    slot->indexFound = found;
                }
                if (slot->indexFound.offset % index.alignment != 0) return refuse(Refusal::NotInPlace);
                index.import = slot->indexFound.import;
                index.handle = slot->indexFound.buffer;
                index.offset = slot->indexFound.offset;
            }
            prepared.ranges.insert(prepared.ranges.end(), slot->ranges.begin(), slot->ranges.end());
            for (const auto& range : prepared.ranges) prepared.inPlaceReads.emplace_back(range.begin, range.end);
            prepared.vertexHandles.assign(slot->handles.begin(), slot->handles.end());
            prepared.vertexOffsets.assign(slot->offsets.begin(), slot->offsets.end());
            prepared.indexBytes = indexBytes;
            prepared.maxIndex = maxIndex;
            if (const auto every = inputsMemoVerifyEvery(); every != 0 && ++memo->ordinal % every == 0) {
                struct FreshStorage {};
                auto& fresh = HostThreadLocal<PreparedInputs, FreshStorage>();
                fresh = {};
                bool made = false;
                try {
                    made = prepareMemoInputs(context, state, draw, shaders, recipe, fresh, nullptr);
                } catch (const std::exception&) {
                    made = false;
                }
                memo->note.verified = true;
                const char* what = made ? preparedInputsDifference(prepared, fresh) : "the refusal";
                fresh = {};
                if (what != nullptr) {
                    memo->note.differed = true;
                    static std::atomic<std::uint32_t> printed{0};
                    if (printed.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[direct-inputs] memo verify: %s differs from today's\n", what);
                    *slot = {};
                    return prepareMemoInputs(context, state, draw, shaders, std::move(recipe), prepared, nullptr);
                }
            }
            prepared.recipe = std::move(recipe);
            return true;
        }
    }
    auto& scratch = HostThreadLocal<PreparedInputScratch, PreparedInputScratch>();
    auto& fetches = scratch.fetches;
    auto& fetchValid = scratch.fetchValid;
    fetches.clear();
    fetchValid.clear();
    for (const auto& attribute : attributes) {
        const auto described = std::find_if(layout.attributes.begin(), layout.attributes.end(), [&](const auto& entry) { return entry.location == attribute.location; });
        Require(described != layout.attributes.end(), "vertex attribute missing from its input layout");
        const auto narrow = DecodeVertexFormat(attribute);
        const std::size_t widened = described->format == narrow.format ? 0u : VertexReadBytes(attribute, described->format) - narrow.bytes;
        if (NullVertexDescriptor(attribute) || VertexFetchOutOfRange(attribute)) return refuse(Refusal::Shape);
        if (args != nullptr && VertexBufferExtent(attribute) == 0) return refuse(Refusal::Shape);
        const auto bytes = (args != nullptr ? VertexBufferExtent(attribute) : VertexBufferReadSize(attribute, maxIndex, draw.instanceCount, draw.firstInstance)) + widened;
        const auto& fields = attribute.resource.fields;
        const auto address = fields[0] | (static_cast<std::uint64_t>(fields[1] & 0xffffu) << 32u);
        const auto firstByte = args == nullptr ? VertexBufferReadOffset(attribute, draw.firstVertex, draw.firstInstance) : 0;
        Require(!state.hasColorTarget || address + bytes <= state.color.MemoryAddress() || state.color.MemoryAddress() + state.color.bytes <= address + firstByte, "vertex buffer aliases the render target");
        const auto stride = (fields[1] >> 16u) & 0x3fffu;
        fetches.push_back({address, address + bytes, stride, attribute.fetchIndex, narrow.alignment});
        const auto recordBytes = static_cast<std::uint64_t>(fields[2]) * stride;
        fetchValid.push_back(args == nullptr && stride != 0 ? static_cast<std::size_t>(std::min(recordBytes, static_cast<std::uint64_t>(bytes))) : bytes);
    }
    SoloZeroPaddedFetchIndicesInto(fetches, fetchValid, scratch.solo);
    if (!scratch.solo.empty()) return refuse(Refusal::Shape);
    PlanVertexCopiesInto(fetches, scratch.plan, scratch.order);
    const auto& plan = scratch.plan;
    for (std::size_t range = 0; range < plan.copies.size(); ++range) {
        const auto [begin, end] = plan.copies[range];
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(begin), static_cast<std::size_t>(end - begin))) return refuse(Refusal::Inaccessible);
        prepared.ranges.push_back({begin, end, plan.alignments[range]});
    }
    if (args != nullptr) {
        const auto recordBytes = args->RangeBytes();
        if (recordBytes > std::numeric_limits<std::size_t>::max() || recordBytes > std::numeric_limits<std::uint64_t>::max() - args->arguments || !GuestMemory::Accessible(reinterpret_cast<const void*>(args->arguments), static_cast<std::size_t>(recordBytes))) return refuse(Refusal::Inaccessible);
        prepared.ranges.push_back({args->arguments, args->arguments + recordBytes, 4u});
        if (args->countIndirect) {
            if (!GuestMemory::Accessible(reinterpret_cast<const void*>(args->countAddress), 4u)) return refuse(Refusal::Inaccessible);
            prepared.ranges.push_back({args->countAddress, args->countAddress + 4u, 4u});
        }
        prepared.indirectRanges = args->countIndirect ? 2u : 1u;
    }
    if (capacity) prepared.inPlaceReads.reserve(prepared.ranges.size());
    const std::size_t first = draw.indexed ? 1u : 0u;
    if (mirrored) {
        // The committer's own choices (prepareDrawInputs): the first byte a vertex copy holds, and a
        // range the draw fetches little of bound in place.
        const auto reads = static_cast<std::uint64_t>(draw.indexCount) * draw.instanceCount;
        const auto starts = VertexCopyStarts(plan, fetches, minIndex, draw.firstInstance);
        for (std::size_t index = 0; index < prepared.ranges.size(); ++index) {
            auto& range = prepared.ranges[index];
            const auto bytes = static_cast<std::size_t>(range.end - range.begin);
            const auto from = index < first ? std::size_t{0} : starts[index - first];
            if (bytes - from <= context.drawInputsInPlaceBelow && reads <= InPlaceDrawInputMaxReads) {
                const std::pair<std::uint64_t, std::uint64_t> read{range.begin, range.end};
                ExistingHostImport existing{};
                if (ExistingHostImportsFor(context, std::span(&read, 1), std::span(&existing, 1)) && existing.offset % range.alignment == 0) {
                    range.import = existing.import;
                    range.handle = existing.buffer;
                    range.offset = existing.offset;
                    range.small = true;
                    prepared.inPlaceReads.push_back(read);
                    continue;
                }
            }
            auto ticket = index < first ? indexTicket : Recorder::DrawInputTicket{};
            auto published = context.recorder->PublishedDrawInputFor(range.begin, bytes, ticket.use, from);
            if (published.buffer == nullptr) return refuse(Refusal::NotInPlace);
            ticket.serial = published.serial;
            ticket.held = published.held;
            ticket.from = from;
            range.handle = published.buffer->Handle();
            range.source = std::move(published.buffer);
            range.ticket = ticket;
        }
        prepared.mirrored = true;
    } else {
        for (const auto& range : prepared.ranges) prepared.inPlaceReads.emplace_back(range.begin, range.end);
        auto& found = scratch.found;
        found.assign(prepared.ranges.size(), {});
        if (!prepared.ranges.empty() && !ExistingHostImportsFor(context, prepared.inPlaceReads, found)) return refuse(Refusal::NotInPlace);
        for (std::size_t index = 0; index < prepared.ranges.size(); ++index) {
            auto& range = prepared.ranges[index];
            if (found[index].offset % range.alignment != 0) return refuse(Refusal::NotInPlace);
            range.import = found[index].import;
            range.handle = found[index].buffer;
            range.offset = found[index].offset;
        }
    }
    prepared.vertexHandles.reserve(attributes.size());
    prepared.vertexOffsets.reserve(attributes.size());
    for (std::size_t index = 0; index < attributes.size(); ++index) {
        const auto& range = prepared.ranges[first + plan.copyOf[index]];
        prepared.vertexHandles.push_back(range.handle);
        prepared.vertexOffsets.push_back(range.offset + plan.offsets[index]);
    }
    prepared.indexBytes = indexBytes;
    prepared.maxIndex = maxIndex;
    if (slot != nullptr) {
        slot->recipe = recipe;
        slot->attributes.assign(attributes.begin(), attributes.end());
        slot->firstVertex = draw.firstVertex;
        slot->instanceCount = draw.instanceCount;
        slot->firstInstance = draw.firstInstance;
        slot->maxIndex = maxIndex;
        slot->indexSize = draw.indexSize;
        slot->indexed = draw.indexed;
        slot->hasColor = state.hasColorTarget;
        slot->colorBegin = state.hasColorTarget ? static_cast<std::uint64_t>(state.color.MemoryAddress()) : 0u;
        slot->colorBytes = state.hasColorTarget ? static_cast<std::uint64_t>(state.color.bytes) : 0u;
        slot->forget = prepared.forget;
        slot->imports = prepared.imports;
        slot->mapping = prepared.mapping;
        slot->ranges.assign(prepared.ranges.begin() + static_cast<std::ptrdiff_t>(first), prepared.ranges.end());
        slot->handles.assign(prepared.vertexHandles.begin(), prepared.vertexHandles.end());
        slot->offsets.assign(prepared.vertexOffsets.begin(), prepared.vertexOffsets.end());
        slot->indexKnown = draw.indexed;
        slot->indexBegin = draw.indexed ? prepared.ranges.front().begin : 0u;
        slot->indexEnd = draw.indexed ? prepared.ranges.front().end : 0u;
        slot->indexFound = draw.indexed ? ExistingHostImport{prepared.ranges.front().import, prepared.ranges.front().handle, prepared.ranges.front().offset} : ExistingHostImport{};
        memo->note.stored = true;
    }
    prepared.recipe = std::move(recipe);
    return true;
}

}

DrawRecordInputs VerifyPreparedInputsForTest(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, const DrawRecipe& recipe, PreparedInputs& prepared, std::string& difference, DrawRecordInputs* taken) {
    PerformanceTimer timing("Graphics.VerifyPreparedInputs");
    DrawTimer timer(false);
    DrawOutcome outcome;
    const auto plan = planDraw(context, state, draw, shaders, &recipe, outcome, timer);
    DrawInputs today;
    if (plan.nothing) today.nothing = true;
    else today = prepareDrawInputs(context, state, draw, shaders, plan, timer, timing, &recipe);
    return verifyPreparedInputs(context, state, draw, shaders, recipe, prepared, today, timer, timing, difference, taken);
}

namespace {

// The resident-target proof of one attachment (StorageTexture::Refresh: FlushPending, CollectWrites
// over the target's pages, the DCC key scan of TextureClearKeys, then UnchangedSince), with the
// [draws] target-lookup accounting. `lookup` makes (or finds) the image; DrawWithRecipe refreshes
// the stored object instead.
std::shared_ptr<StorageTexture> refreshResidentTarget(const Context& context, const State& state, const ColorTarget& color, DrawOutcome& outcome, bool profile, const std::function<std::shared_ptr<StorageTexture>()>& lookup) {
    const auto lookupStart = std::chrono::steady_clock::now();
    const auto walkedBefore = profile ? GuestMemory::ThreadCollectedBytes() : 0;
    std::shared_ptr<StorageTexture> resident;
    try {
        resident = lookup();
        if (resident != nullptr) materializeRegisterClear(context, color, *resident);
    } catch (const std::exception& error) {
        resident = nullptr;
        static AgcDriver::Mutex reportMutex;
        static std::set<std::uint64_t> reported;
        std::lock_guard lock(reportMutex);
        if (reported.insert(color.address).second) std::fprintf(stderr, "[gpu] color target 0x%llx stays non-resident: %s\n", static_cast<unsigned long long>(color.address), error.what());
    }
    if (resident != nullptr) materializeCmaskClear(context, color, resident);
    if (profile) {
        const auto lookupUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - lookupStart).count();
        ++outcome.targetLookups;
        if (lookupUs >= 1000.0) {
            ++outcome.slowLookups;
            outcome.slowLookupUs += lookupUs;
        }
        outcome.lookupUs += lookupUs;
        outcome.pagesWalked += (GuestMemory::ThreadCollectedBytes() - walkedBefore) / 4096;
        const bool fullScissor = state.scissor.offset.x <= 0 && state.scissor.offset.y <= 0 && state.scissor.extent.width >= color.extent.width && state.scissor.extent.height >= color.extent.height;
        ++(fullScissor ? outcome.fullScissorLookups : outcome.partialLookups);
    }
    return resident;
}

// A draw's resources (resolveDrawResources): the resource-cache template a recordable draw's
// stages repeat, or a fresh build.
struct ResolvedResources {
    std::shared_ptr<ShaderResources> resources;
    std::vector<ShaderResources::MovedBuffer> moved;
    ResourceCache::Key contentKey;
    bool cacheable = false;
    bool replay = false;
    const ShaderResources::BuildTiming* built = nullptr;
};

enum class AddressStages { None, ReadOnly, Storing };

AddressStages addressStages(std::span<const CompiledShader> shaders) {
    auto stages = AddressStages::None;
    for (const auto& shader : shaders) {
        if (shader.program == nullptr || std::none_of(shader.program->bindings.begin(), shader.program->bindings.end(), [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable; })) continue;
        if (shader.program->bdaStores) return AddressStages::Storing;
        stages = AddressStages::ReadOnly;
    }
    return stages;
}

bool verifyAddressDrawReuse() {
    static const bool verify = std::getenv("APS5_VERIFY_ADDRESS_DRAW_REUSE") != nullptr;
    return verify;
}

bool snapshotsInTable(std::span<const std::byte> table, std::span<const GuestMemorySnapshot> snapshots) {
    ShaderRecompiler::BdaAbi::Header header{};
    if (table.size() < sizeof(header)) return snapshots.empty();
    std::memcpy(&header, table.data(), sizeof(header));
    if (header.entryBytes != sizeof(ShaderRecompiler::BdaAbi::Range) || (table.size() - sizeof(header)) / sizeof(ShaderRecompiler::BdaAbi::Range) < header.count) return false;
    std::vector<ShaderRecompiler::BdaAbi::Range> ranges(header.count);
    if (!ranges.empty()) std::memcpy(ranges.data(), table.data() + sizeof(header), ranges.size() * sizeof(ranges.front()));
    return std::ranges::all_of(snapshots, [&](const GuestMemorySnapshot& snapshot) {
        auto at = snapshot.address;
        const auto end = snapshot.address + snapshot.bytes.size();
        for (const auto& range : ranges) {
            if (at >= end || range.begin > at) break;
            if (range.end > at && (range.permissions & ShaderRecompiler::BdaAbi::Read) != 0u) at = range.end;
        }
        return at >= end;
    });
}

bool snapshotsEqualMemory(std::span<const GuestMemorySnapshot> snapshots) {
    return std::ranges::all_of(snapshots, [](const GuestMemorySnapshot& snapshot) { return snapshot.bytes.empty() || std::memcmp(reinterpret_cast<const void*>(snapshot.address), snapshot.bytes.data(), snapshot.bytes.size()) == 0; });
}

void verifyServedSpace(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, std::uint64_t indexBytes, const ShaderResources& served) {
    const ShaderResources fresh(context, shaders, state.color, static_cast<std::uint32_t>(state.colors.size()), draw.indexAddress, static_cast<std::size_t>(indexBytes), snapshots);
    const auto servedTable = served.AddressTableBytes();
    const auto freshTable = fresh.AddressTableBytes();
    const char* what = !fresh.SpaceReuse() ? "space reuse" : !std::ranges::equal(servedTable, freshTable) ? "BDA table" : !snapshotsInTable(servedTable, snapshots) ? "snapshot ranges" : !snapshotsEqualMemory(snapshots) ? "snapshot bytes" : served.SampledTextures() != fresh.SampledTextures() ? "sampled textures" : served.StorageImages() != fresh.StorageImages() ? "storage images" : served.LayoutKey() != fresh.LayoutKey() || served.FragmentLayoutKey() != fresh.FragmentLayoutKey() ? "set layouts" : nullptr;
    addressDrawReuseVerifications().fetch_add(1, std::memory_order_relaxed);
    if (what == nullptr) return;
    char message[320];
    std::snprintf(message, sizeof(message), "APS5_VERIFY_ADDRESS_DRAW_REUSE: a served address-based draw hit disagrees with a fresh build or guest memory in its %s (BDA table %zu bytes served, %zu fresh; %zu / %zu sampled textures; %zu snapshots)", what, servedTable.size(), freshTable.size(), served.SampledTextures().size(), fresh.SampledTextures().size(), snapshots.size());
    throw std::runtime_error(message);
}

std::uint64_t movedInPlaceVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_MOVED_IN_PLACE");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

void verifyMovedInPlace(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::uint64_t indexBytes, const ShaderResources& served, std::span<const ShaderResources::MovedBuffer> moved) {
    const auto every = movedInPlaceVerifyEvery();
    if (every == 0 || std::ranges::none_of(moved, [](const ShaderResources::MovedBuffer& entry) { return entry.inPlace != VK_NULL_HANDLE; })) return;
    static std::atomic<std::uint64_t> ordinal{0};
    if (ordinal.fetch_add(1, std::memory_order_relaxed) % every != 0) return;
    std::vector<ShaderResources::BoundBuffer> mine;
    served.DescribeMovedInPlace(moved, mine);
    std::vector<ShaderResources::BoundBuffer> rebuilt;
    try {
        auto fresh = std::make_shared<ShaderResources>(context, shaders, state.color, static_cast<std::uint32_t>(state.colors.size()), draw.indexAddress, static_cast<std::size_t>(indexBytes));
        if (auto* recorder = Recorder::Active(); recorder != nullptr) recorder->Keep(fresh);
        fresh->DescribeDrawBindings(nullptr, {}, nullptr, rebuilt);
    } catch (const std::exception& error) {
        static std::atomic<std::uint64_t> notRun{0};
        const auto skipped = notRun.fetch_add(1, std::memory_order_relaxed) + 1u;
        if (skipped <= 16u || skipped % 4096u == 0u) std::fprintf(stderr, "[moved-in-place-verify] %llu checks not run: the rebuild threw: %s\n", static_cast<unsigned long long>(skipped), error.what());
        return;
    }
    static std::atomic<std::uint64_t> checks{0}, differing{0}, printed{0};
    const auto checked = checks.fetch_add(1, std::memory_order_relaxed) + 1u;
    bool differs = false;
    for (const auto& entry : mine) {
        const auto found = std::ranges::find_if(rebuilt, [&](const ShaderResources::BoundBuffer& other) { return other.set == entry.set && other.binding == entry.binding && other.element == entry.element; });
        if (found != rebuilt.end() && !found->snapshot && found->info.buffer == entry.info.buffer && found->info.offset == entry.info.offset && found->info.range == entry.info.range) continue;
        differs = true;
        if (printed.fetch_add(1, std::memory_order_relaxed) >= 16u) continue;
        if (found == rebuilt.end()) {
            std::fprintf(stderr, "[moved-in-place-verify] set %u binding %u[%u], 0x%llx+0x%llx bound in place at offset %llu, range %llu: the rebuild has no such binding\n", entry.set, entry.binding, entry.element, static_cast<unsigned long long>(entry.address), static_cast<unsigned long long>(entry.bytes), static_cast<unsigned long long>(entry.info.offset), static_cast<unsigned long long>(entry.info.range));
            continue;
        }
        const char* other = found->snapshot ? "a snapshot of" : found->info.buffer == entry.info.buffer ? "the same import as" : "another buffer than";
        std::fprintf(stderr, "[moved-in-place-verify] set %u binding %u[%u], 0x%llx+0x%llx bound in place at offset %llu, range %llu: the rebuild binds %s the in-place one at offset %llu, range %llu\n", entry.set, entry.binding, entry.element, static_cast<unsigned long long>(entry.address), static_cast<unsigned long long>(entry.bytes), static_cast<unsigned long long>(entry.info.offset), static_cast<unsigned long long>(entry.info.range), other, static_cast<unsigned long long>(found->info.offset), static_cast<unsigned long long>(found->info.range));
    }
    const auto different = differs ? differing.fetch_add(1, std::memory_order_relaxed) + 1u : differing.load(std::memory_order_relaxed);
    if (differs || checked % 4096u == 0u) std::fprintf(stderr, "[moved-in-place-verify] %llu checked, %llu differing\n", static_cast<unsigned long long>(checked), static_cast<unsigned long long>(different));
}

void noteKeptHit(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, std::uint64_t indexBytes, const ShaderResources& served, std::span<const ShaderResources::MovedBuffer> moved, bool refilled) {
    auto& census = templateKeepDrawCensus();
    if (refilled) census.keptHits.fetch_add(1, std::memory_order_relaxed);
    const auto every = templateKeepsVerifyEvery();
    static std::atomic<std::uint64_t> ordinal{0};
    if (every == 0 || ordinal.fetch_add(1, std::memory_order_relaxed) % every != 0) return;
    std::shared_ptr<ShaderResources> fresh;
    try {
        fresh = std::make_shared<ShaderResources>(context, shaders, state.color, static_cast<std::uint32_t>(state.colors.size()), draw.indexAddress, static_cast<std::size_t>(indexBytes), snapshots);
        if (auto* recorder = Recorder::Active(); recorder != nullptr) recorder->Keep(fresh);
    } catch (const std::exception& error) {
        census.notRun.fetch_add(1, std::memory_order_relaxed);
        static std::atomic<std::uint64_t> notRun{0};
        const auto skipped = notRun.fetch_add(1, std::memory_order_relaxed) + 1u;
        if (skipped <= 16u || skipped % 4096u == 0u) std::fprintf(stderr, "[template-keeps-verify] %llu checks of kept templates not run: the fresh build threw: %s\n", static_cast<unsigned long long>(skipped), error.what());
        return;
    }
    const char* what = served.KeptDifference(*fresh, moved);
    static std::atomic<std::uint64_t> checks{0};
    static std::atomic<std::uint64_t> differences{0};
    census.verified.fetch_add(1, std::memory_order_relaxed);
    if (what != nullptr) census.differing.fetch_add(1, std::memory_order_relaxed);
    const auto checked = checks.fetch_add(1, std::memory_order_relaxed) + 1u;
    const auto differing = what != nullptr ? differences.fetch_add(1, std::memory_order_relaxed) + 1u : differences.load(std::memory_order_relaxed);
    if (what != nullptr && (differing <= 16u || differing % 4096u == 0u)) std::fprintf(stderr, "[template-keeps-verify] %s differs from a fresh build in its %s; %llu kept uses checked, %llu differing\n", refilled ? "a template kept over a refilled texture" : "a rect-list template keyed by the generated-stage rule", what, static_cast<unsigned long long>(checked), static_cast<unsigned long long>(differing));
}

void noteRevivedHit(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, std::uint64_t indexBytes, const ShaderResources& served, std::span<const ShaderResources::MovedBuffer> moved) {
    auto& census = viewKeepDrawCensus();
    if (ResourceCache::FateCensus()) census.revivedHits.fetch_add(1, std::memory_order_relaxed);
    const auto every = viewTemplateKeepsVerifyEvery();
    static std::atomic<std::uint64_t> ordinal{0};
    if (every == 0 || ordinal.fetch_add(1, std::memory_order_relaxed) % every != 0) return;
    std::shared_ptr<ShaderResources> fresh;
    try {
        fresh = std::make_shared<ShaderResources>(context, shaders, state.color, static_cast<std::uint32_t>(state.colors.size()), draw.indexAddress, static_cast<std::size_t>(indexBytes), snapshots);
        if (auto* recorder = Recorder::Active(); recorder != nullptr) recorder->Keep(fresh);
    } catch (const std::exception& error) {
        census.notRun.fetch_add(1, std::memory_order_relaxed);
        static std::atomic<std::uint64_t> notRun{0};
        const auto skipped = notRun.fetch_add(1, std::memory_order_relaxed) + 1u;
        if (skipped <= 16u || skipped % 4096u == 0u) std::fprintf(stderr, "[view-template-keeps-verify] %llu checks of templates kept over a taken-back texture not run: the fresh build threw: %s\n", static_cast<unsigned long long>(skipped), error.what());
        return;
    }
    const char* what = served.KeptDifference(*fresh, moved);
    static std::atomic<std::uint64_t> checks{0};
    static std::atomic<std::uint64_t> differences{0};
    census.verified.fetch_add(1, std::memory_order_relaxed);
    if (what != nullptr) census.differing.fetch_add(1, std::memory_order_relaxed);
    const auto checked = checks.fetch_add(1, std::memory_order_relaxed) + 1u;
    const auto differing = what != nullptr ? differences.fetch_add(1, std::memory_order_relaxed) + 1u : differences.load(std::memory_order_relaxed);
    if (what != nullptr && (differing <= 16u || differing % 4096u == 0u)) std::fprintf(stderr, "[view-template-keeps-verify] a template kept over a taken-back texture differs from a fresh build in its %s; %llu kept uses checked, %llu differing\n", what, static_cast<unsigned long long>(checked), static_cast<unsigned long long>(differing));
}

void noteLeanCoverage(const ShaderResources& resources, std::span<const ShaderResources::MovedBuffer> moved, const Recorder* recorder) {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    if (!stats) return;
    enum : std::size_t { Hits, BackToBack, SameBatch, Moved0, Moved1, Moved2, Moved3, MovedBytes, Count };
    static std::array<std::atomic<std::uint64_t>, Count> counts{};
    static std::atomic<std::int64_t> reportedAt{0};
    thread_local const ShaderResources* previous = nullptr;
    counts[Hits].fetch_add(1, std::memory_order_relaxed);
    if (previous == &resources) counts[BackToBack].fetch_add(1, std::memory_order_relaxed);
    previous = &resources;
    const auto serial = recorder != nullptr ? recorder->OpenSerial() : 0;
    if (serial != 0 && resources.CoverageSerial() == serial) counts[SameBatch].fetch_add(1, std::memory_order_relaxed);
    resources.NoteCoverageSerial(serial);
    counts[Moved0 + std::min<std::size_t>(moved.size(), 3)].fetch_add(1, std::memory_order_relaxed);
    std::uint64_t bytes = 0;
    for (const auto& entry : moved) bytes += entry.size;
    counts[MovedBytes].fetch_add(bytes, std::memory_order_relaxed);
    const auto tenSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (reportedAt.exchange(tenSeconds, std::memory_order_relaxed) == tenSeconds) return;
    std::array<unsigned long long, Count> taken{};
    for (std::size_t index = 0; index < Count; ++index) taken[index] = counts[index].exchange(0, std::memory_order_relaxed);
    const auto snapshots = ShaderResources::TakeMovedSnapshotCounts();
    std::fprintf(stderr, "[lean-coverage] template hits (10 s): %llu; same template as the previous draw %llu; template already used in the open batch %llu; moved buffers per hit 0: %llu, 1: %llu, 2: %llu, 3+: %llu; moved bytes %llu; moved snapshots up to 64 KiB through the snapshot cache %llu (%llu reused, %llu bytes), copied into the upload chunk %llu (%llu bytes)\n", taken[Hits], taken[BackToBack], taken[SameBatch], taken[Moved0], taken[Moved1], taken[Moved2], taken[Moved3], taken[MovedBytes], static_cast<unsigned long long>(snapshots.cached), static_cast<unsigned long long>(snapshots.reused), static_cast<unsigned long long>(snapshots.cachedBytes), static_cast<unsigned long long>(snapshots.uploaded), static_cast<unsigned long long>(snapshots.uploadedBytes));
    bool anyMoved = false;
    for (std::size_t arm = 0; arm < ShaderResources::MovedCountArms; ++arm) {
        const auto movedCounts = ShaderResources::TakeMovedCounts(arm);
        std::string movedText;
        for (std::size_t index = 0; index < movedCounts.size(); ++index) {
            if (movedCounts[index] == 0) continue;
            movedText += movedText.empty() ? " " : ", ";
            movedText += ShaderResources::MovedCountName(static_cast<ShaderResources::MovedCount>(index));
            movedText += ' ';
            movedText += std::to_string(movedCounts[index]);
        }
        if (movedText.empty()) continue;
        anyMoved = true;
        std::fprintf(stderr, "[lean-coverage] moved buffers (10 s, counted while APS5_MOVED_IN_PLACE was %s and APS5_DIRECT_BUFFERS_IN_PLACE %s):%s\n", (arm & 1u) != 0 ? "set" : "unset", (arm & 2u) != 0 ? "set" : "unset", movedText.c_str());
    }
    if (!anyMoved) std::fprintf(stderr, "[lean-coverage] moved buffers (10 s): none\n");
}

bool drawStagingEnabled() {
    static const bool disabled = std::getenv("APS5_NO_DRAW_STAGING") != nullptr;
    return !disabled;
}

ResolvedResources resolveDrawResources(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, std::uint64_t indexBytes, bool recordable, const DrawPlan& plan, DrawOutcome& outcome, DrawTimer& timer, ReplayReason replayDraw, bool replayOn, DrawCommitView* view = nullptr) {
    ResolvedResources resolved;
    APS5_LOG_CHARS_OUT_DEBUG("Creating ShaderResources");
    // A recordable draw whose stages' compiled content repeats an earlier one binds that build's
    // descriptor set when it is still valid (see ResourceCache; the dispatch path does the same).
    // Only recordable draws take part: a synchronous draw writes its resources back, which a shared
    // object must never do. Revalidate proves a hit by texture identity, which needs the texture
    // caches. Push constants still come from this draw's stages. APS5_NO_DRAW_RESOURCE_CACHE=1
    // builds every draw's resources as before.
    // The render target and index ranges stay out of the key (see DrawResourceKey); a hit repeats
    // the alias checks instead. Debug aid: APS5_NO_DRAW_KEY_TRIM=1 keys them as before.
    static const bool trimKey = std::getenv("APS5_NO_DRAW_KEY_TRIM") == nullptr;
    const bool candidate = recordable && plan.candidate;
    HeapCostScope findCost(HeapCost::TemplateFind);
    resolved.cacheable = recordable && plan.keyed;
    const auto address = candidate && !ShaderResources::SpaceReuseDisabled() ? addressStages(shaders) : AddressStages::None;
    std::optional<AddressDrawReuse> reuse;
    if (address == AddressStages::Storing) reuse = AddressDrawReuse::NotReadOnly;
    else if (address == AddressStages::ReadOnly && !resolved.cacheable) reuse = AddressDrawReuse::NotSpaceOnly;
    auto buildCause = TemplateBuildCause::NotKeyed;
    if (resolved.cacheable) {
        buildCause = TemplateBuildCause::NoEntry;
        resolved.contentKey = plan.contentKey;
        if (view != nullptr) view->lookup = DrawCommitLookup::Absent;
        if (auto cached = SharedResourceCache().Find(resolved.contentKey)) {
            timer.lap(FullDrawStep::Lookup);
            if (view != nullptr) view->found = cached.get();
            const auto releaseSpace = [](ShaderResources* resources) { static_cast<void>(resources->ReleaseSpace()); };
            std::unique_ptr<ShaderResources, decltype(releaseSpace)> heldSpace(nullptr, releaseSpace);
            const bool counted = address == AddressStages::ReadOnly && cached->SpaceReuse();
            bool spaceHeld = !cached->SpaceReuse();
            bool inUse = false;
            if (!spaceHeld) {
                if (auto* active = Recorder::Active(); active != nullptr) {
                    GuestBufferMemory::ReleaseRetainedSpaces(active->Id(), active->CompletedThrough());
                    static_cast<void>(cached->ReleaseCompletedSpace(active->Id(), active->CompletedThrough()));
                }
                if (cached->SpaceHeld()) {
                    inUse = true;
                    reuse = AddressDrawReuse::InUse;
                } else if (!cached->RetakeSpace()) {
                    reuse = AddressDrawReuse::SpaceChanged;
                } else {
                    heldSpace.reset(cached.get());
                    if (!cached->AddressTableGuardCurrent()) {
                        reuse = AddressDrawReuse::SpaceChanged;
                        static_cast<void>(cached->ReleaseSpace());
                    } else {
                        completeSnapshotWrites(snapshots);
                        const auto service = cached->SnapshotsServedInPlace(snapshots);
                        spaceHeld = service == SnapshotService::Served;
                        if (!spaceHeld) {
                            reuse = service == SnapshotService::Partial ? AddressDrawReuse::PartialOverlap : service == SnapshotService::Changed ? AddressDrawReuse::SnapshotChanged : AddressDrawReuse::SnapshotOutside;
                            static_cast<void>(cached->ReleaseSpace());
                        }
                    }
                }
            }
            timer.lap(FullDrawStep::Space);
            ShaderResources::ProofReport proof;
            const bool valid = spaceHeld && (RuntimeSwitches::Set(RuntimeSwitch::XpProofAsIs) ? cached->AsIsCurrent() : cached->Revalidate(shaders, &proof));
            timer.lap(FullDrawStep::Proof);
            if (counted && spaceHeld && !valid) reuse = AddressDrawReuse::Revalidation;
            auto* recorder = Recorder::Active();
            std::optional<std::vector<ShaderResources::MovedBuffer>> moved;
            bool patched = false;
            if (valid && recorder != nullptr) moved = cached->MovedReadOnlyBuffers(shaders, *recorder, &state.color, &patched);
            if (moved.has_value()) {
                if (trimKey) CheckBufferAliases(shaders, state.color, draw.indexAddress, indexBytes);
                if (counted && verifyAddressDrawReuse()) verifyServedSpace(context, state, draw, shaders, snapshots, indexBytes, *cached);
                static_cast<void>(heldSpace.release());
                resolved.resources = std::move(cached);
                resolved.moved = std::move(*moved);
                noteLeanCoverage(*resolved.resources, resolved.moved, recorder);
                const bool keptUse = templateKeepsCensus() && resolved.resources->KeptUse();
                if (keptUse || (plan.generatedKey && templateKeepsVerifyEvery() != 0)) noteKeptHit(context, state, draw, shaders, snapshots, indexBytes, *resolved.resources, resolved.moved, keptUse);
                if (viewTemplateKeepsWatched() && resolved.resources->RevivedUse()) noteRevivedHit(context, state, draw, shaders, snapshots, indexBytes, *resolved.resources, resolved.moved);
                if (plan.generatedKey && ResourceCache::FateCensus()) templateKeepDrawCensus().generatedHits.fetch_add(1, std::memory_order_relaxed);
                if (resolved.resources->HeapStage()) {
                    const auto reason = replayDraw == ReplayReason::Eligible ? resolved.resources->HeapReplayRefusal(shaders) : replayDraw;
                    if (ReplayCensusEnabled() && replayDraw != ReplayReason::Count) {
                        NoteReplayReason(reason);
                        resolved.resources->NoteHeapReplayCensus();
                    }
                    resolved.replay = replayOn && reason == ReplayReason::Eligible;
                    if (!resolved.replay) resolved.resources->RefreshHeapStage(shaders);
                }
                outcome.addressBased = resolved.resources->HoldsLease();
                outcome.kind = KindTemplateHit;
                countCache(&DrawProfile::cacheHits);
                if (counted) reuse = AddressDrawReuse::Reused;
                if (ResourceCache::FateCensus()) noteTemplateLookup(resolved.contentKey, buildCause, false);
                if (view != nullptr) view->lookup = DrawCommitLookup::Hit;
            } else if (valid || inUse) {
                buildCause = !valid ? TemplateBuildCause::InUse : recorder == nullptr ? TemplateBuildCause::NoRecorder : patched ? TemplateBuildCause::PatchedData : TemplateBuildCause::UnmovableBuffer;
                if (view != nullptr) view->lookup = valid ? DrawCommitLookup::Unmoved : DrawCommitLookup::Invalid;
                if (valid && cached->SpaceReuse()) static_cast<void>(cached->ReleaseSpace());
                if (valid && counted) reuse = patched ? AddressDrawReuse::PatchedData : AddressDrawReuse::MovedBuffer;
                countCache(&DrawProfile::cacheMisses);
            } else {
                if (view != nullptr) view->lookup = DrawCommitLookup::Invalid;
                if (cached->SpaceReuse()) static_cast<void>(cached->ReleaseSpace());
                SharedResourceCache().Remove(resolved.contentKey);
                countCache(&DrawProfile::cacheInvalidated);
                buildCause = spaceHeld ? TemplateBuildCause::Proof : TemplateBuildCause::SpaceNotServed;
                if (!spaceHeld) ResourceCache::NoteFate(resolved.contentKey, ResourceCache::Fate::LookupSpace);
                else noteFailedProof(context, ProofSite::Lookup, proof, &resolved.contentKey);
            }
        }
    } else {
        if (view != nullptr) view->lookup = DrawCommitLookup::Uncacheable;
        countCache(&DrawProfile::uncacheable);
        if (recordable && ResourceCache::FateCensus()) noteNotKeyed(notKeyedReason(state, shaders));
    }
    findCost.Stop();
    timer.phase(PhaseLookup);
    if (resolved.resources == nullptr) {
        if (ResourceCache::FateCensus()) noteTemplateLookup(resolved.contentKey, buildCause, true);
        if (plan.generatedKey && ResourceCache::FateCensus()) templateKeepDrawCensus().generatedBuilt.fetch_add(1, std::memory_order_relaxed);
        HeapCostScope buildCost(HeapCost::TemplateBuild);
        resolved.resources = std::make_shared<ShaderResources>(context, shaders, state.color, static_cast<std::uint32_t>(state.colors.size()), draw.indexAddress, static_cast<std::size_t>(indexBytes), snapshots, recordable && drawStagingEnabled());
        buildCost.Stop();
        resolved.built = &resolved.resources->Timing();
        outcome.addressBased = resolved.resources->HoldsLease();
        outcome.kind = outcome.addressBased ? KindBda : KindBuild;
        if (resolved.cacheable) countCache(&DrawProfile::cacheMisses);
        if (address == AddressStages::ReadOnly && !reuse.has_value()) reuse = resolved.resources->SpaceReuse() ? AddressDrawReuse::Built : AddressDrawReuse::NotSpaceOnly;
    }
    if (reuse.has_value()) countAddressDrawReuse(*reuse);
    timer.phase(PhaseResources);
    timer.lap(FullDrawStep::Moved);
    APS5_LOG_CHARS_OUT_DEBUG("ShaderResources created");
    return resolved;
}

// A GPU-side indirect draw's records (Draw's recorded path; a recipe draw has none) as recordDraw
// takes them: the arguments, how they are read, their host imports and, on the CPU path, the
// records read.
struct IndirectRecord {
    const Pm4::DrawParameters::IndirectDraw* args = nullptr;
    IndirectDrawPath path = IndirectDrawPath::Gpu;
    const HostImport* argumentImport = nullptr;
    const HostImport* countImport = nullptr;
    std::span<const Pm4::DrawArguments> records;
    double readMs = 0;
};

void CompactIndirectListRestart(const Context& context, const Pm4::DrawParameters& draw, DrawInputs& inputs, std::vector<Pm4::DrawArguments>& records) {
    const auto bound = inputs.indexCount.value_or(draw.indexCount);
    const auto raw = inputs.indices->Bytes();
    std::uint64_t total = 0;
    for (const auto& record : records) {
        if (record.count != 0 && record.instances != 0 && record.firstVertexOrIndex < bound) total += std::min(record.count, bound - record.firstVertexOrIndex);
    }
    Require(total <= std::numeric_limits<std::uint32_t>::max(), "indirect list restart index count overflow");
    auto compacted = std::make_shared<Buffer>(context, static_cast<std::size_t>(std::max<std::uint64_t>(total, 1u) * draw.indexSize), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    const auto out = compacted->Bytes();
    std::uint32_t cursor = 0;
    for (auto& record : records) {
        if (record.count == 0 || record.instances == 0 || record.firstVertexOrIndex >= bound) {
            record.count = 0;
            continue;
        }
        const auto count = std::min(record.count, bound - record.firstVertexOrIndex);
        const auto at = out.subspan(static_cast<std::size_t>(cursor) * draw.indexSize, static_cast<std::size_t>(count) * draw.indexSize);
        std::memcpy(at.data(), raw.data() + static_cast<std::size_t>(record.firstVertexOrIndex) * draw.indexSize, at.size());
        record.count = CompactListRestart(at, draw.indexSize, inputs.listRestartPrimitive);
        record.firstVertexOrIndex = cursor;
        cursor += record.count;
    }
    inputs.indexCount = cursor;
    inputs.indices = std::move(compacted);
    inputs.indexHandle = inputs.indices->Handle();
    inputs.indexOffset = 0;
}

// The draw commands of one draw: the vertex and index buffer binds, then the direct draw, the
// GPU-side indirect draw from `argumentBuffer` or the CPU-read records with the driver's rules.
bool recordDrawCommandsImpl(const Context& context, VkCommandBuffer commands, const State& state, const Pm4::DrawParameters& draw, const DrawInputs& inputs, const IndirectRecord* indirect, VkBuffer argumentBuffer, VkDeviceSize argumentOffset, const DrawTraceContext* trace, std::uint32_t& subdraw, std::optional<std::uint32_t> pieceFirstIndex = std::nullopt) {
    bool emitted = false;
    if (context.recorder != nullptr) context.recorder->NoteSampledDraw(commands);
    const auto recorded = [&](std::uint32_t index) {
        if (trace != nullptr && context.recorder != nullptr) {
            context.recorder->NoteDrawTraceToken(trace->token);
            CaptureTrace::Log("draw-recorded token=%llu batch=%llu submission=%llu queue=%u offset=%zu target=%llx subdraw=%u", static_cast<unsigned long long>(trace->token), static_cast<unsigned long long>(context.recorder->OpenBatchSerial()), static_cast<unsigned long long>(trace->submission), trace->queue, trace->offset, static_cast<unsigned long long>(trace->target), index);
        }
    };
    const auto* args = indirect != nullptr ? indirect->args : nullptr;
    if (state.stages.mesh && (args != nullptr || state.tessellationCapture != nullptr)) {
        CountVkDraw(VkDrawKind::Mesh);
        context.Function<PFN_vkCmdDrawMeshTasksIndirectEXT>("vkCmdDrawMeshTasksIndirectEXT")(commands, argumentBuffer, argumentOffset, 1, sizeof(VkDrawMeshTasksIndirectCommandEXT));
        recorded(subdraw++);
        emitted = true;
        return true;
    }
    if (state.stages.mesh) {
        APS5_LOG_OUT_DEBUG("vkCmdDrawMeshTasksEXT groups=%u instances=%u layers=%u", inputs.meshGroups, draw.instanceCount, inputs.meshLayers);
        CountVkDraw(VkDrawKind::Mesh);
        context.Function<PFN_vkCmdDrawMeshTasksEXT>("vkCmdDrawMeshTasksEXT")(commands, inputs.meshGroups, draw.instanceCount, inputs.meshLayers);
        recorded(subdraw++);
        emitted = true;
        return true;
    }
    if (!inputs.vertexHandles.empty()) context.Resolved(&DeviceFunctions::cmdBindVertexBuffers, "vkCmdBindVertexBuffers")(commands, 0, static_cast<std::uint32_t>(inputs.vertexHandles.size()), inputs.vertexHandles.data(), inputs.vertexOffsets.data());
    if (inputs.expandedIndexCount != 0u) {
        context.Resolved(&DeviceFunctions::cmdBindIndexBuffer, "vkCmdBindIndexBuffer")(commands, inputs.indexHandle, inputs.indexOffset, VK_INDEX_TYPE_UINT32);
        CountVkDraw(VkDrawKind::Indexed);
        context.Resolved(&DeviceFunctions::cmdDrawIndexed, "vkCmdDrawIndexed")(commands, inputs.expandedIndexCount, draw.instanceCount, 0, static_cast<std::int32_t>(draw.firstVertex), draw.firstInstance);
        emitted = true;
        return true;
    }
    if (draw.indexed) context.Resolved(&DeviceFunctions::cmdBindIndexBuffer, "vkCmdBindIndexBuffer")(commands, inputs.indexHandle, inputs.indexOffset, draw.indexSize == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    if (args == nullptr) {
        if (draw.indexed) {
            CountVkDraw(VkDrawKind::Indexed);
            context.Resolved(&DeviceFunctions::cmdDrawIndexed, "vkCmdDrawIndexed")(commands, pieceFirstIndex ? draw.indexCount : inputs.indexCount.value_or(draw.indexCount), draw.instanceCount, pieceFirstIndex.value_or(0u), static_cast<std::int32_t>(draw.firstVertex), draw.firstInstance);
            recorded(subdraw++);
            emitted = true;
        } else {
            CountVkDraw(VkDrawKind::Direct);
            context.Resolved(&DeviceFunctions::cmdDraw, "vkCmdDraw")(commands, draw.indexCount, draw.instanceCount, draw.firstVertex, draw.firstInstance);
            recorded(subdraw++);
            emitted = true;
        }
    } else if (indirect->path == IndirectDrawPath::Gpu) {
        emitted = true;
        if (args->countIndirect) {
            CountVkDraw(VkDrawKind::Indirect);
            if (draw.indexed) context.Function<PFN_vkCmdDrawIndexedIndirectCountKHR>("vkCmdDrawIndexedIndirectCountKHR")(commands, argumentBuffer, argumentOffset, indirect->countImport->buffer, HostImportOffset(*indirect->countImport, args->countAddress), args->count, args->stride);
            else context.Function<PFN_vkCmdDrawIndirectCountKHR>("vkCmdDrawIndirectCountKHR")(commands, argumentBuffer, argumentOffset, indirect->countImport->buffer, HostImportOffset(*indirect->countImport, args->countAddress), args->count, args->stride);
        } else if (args->count <= 1 || context.multiDrawIndirect) {
            CountVkDraw(VkDrawKind::Indirect);
            if (draw.indexed) context.Resolved(&DeviceFunctions::cmdDrawIndexedIndirect, "vkCmdDrawIndexedIndirect")(commands, argumentBuffer, argumentOffset, args->count, args->stride);
            else context.Resolved(&DeviceFunctions::cmdDrawIndirect, "vkCmdDrawIndirect")(commands, argumentBuffer, argumentOffset, args->count, args->stride);
        } else {
            for (std::uint32_t record = 0; record < args->count; ++record) {
                const auto offset = argumentOffset + static_cast<VkDeviceSize>(record) * args->stride;
                CountVkDraw(VkDrawKind::Indirect);
                if (draw.indexed) context.Resolved(&DeviceFunctions::cmdDrawIndexedIndirect, "vkCmdDrawIndexedIndirect")(commands, argumentBuffer, offset, 1, args->stride);
                else context.Resolved(&DeviceFunctions::cmdDrawIndirect, "vkCmdDrawIndirect")(commands, argumentBuffer, offset, 1, args->stride);
            }
        }
    } else {
        // The CPU-read records with the driver's rules: an InPlace dimension takes the record's
        // value, a Constant one (the CP writes nowhere) the constant.
        using Rule = Pm4::DrawParameters::IndirectDraw::Rule;
        for (const auto& record : indirect->records) {
            if (record.count == 0 || record.instances == 0) continue;
            const auto firstInstance = args->instanceRule == Rule::InPlace ? record.firstInstance : args->instanceConstant;
            if (draw.indexed) {
                // The CP clamps the index range to INDEX_BUFFER_SIZE (the bound copy).
                const auto boundCount = inputs.indexCount.value_or(draw.indexCount);
                if (record.firstVertexOrIndex >= boundCount) continue;
                const auto vertexOffset = args->vertexRule == Rule::InPlace ? record.vertexOffset : args->vertexConstant;
                emitted = true;
                CountVkDraw(VkDrawKind::Indexed);
                context.Resolved(&DeviceFunctions::cmdDrawIndexed, "vkCmdDrawIndexed")(commands, std::min(record.count, boundCount - record.firstVertexOrIndex), record.instances, record.firstVertexOrIndex, static_cast<std::int32_t>(vertexOffset), firstInstance);
            } else {
                const auto firstVertex = args->vertexRule == Rule::InPlace ? record.firstVertexOrIndex : args->vertexConstant;
                emitted = true;
                CountVkDraw(VkDrawKind::Direct);
                context.Resolved(&DeviceFunctions::cmdDraw, "vkCmdDraw")(commands, record.count, record.instances, firstVertex, firstInstance);
            }
        }
    }
    return emitted;
}


void NoteGuestWorkIn(const Context& context, VkCommandBuffer commands) {
    static const bool reporting = LibcRunReportEnabled_nid_no_patch();
    if (!reporting) return;
    if (context.recorder != nullptr && context.recorder->NoteGuestWork(commands)) return;
    NoteGuestWorkOnCommandBuffer(commands);
}

void recordDrawCommands(const Context& context, VkCommandBuffer commands, const State& state, const Pm4::DrawParameters& draw, const DrawInputs& inputs, const IndirectRecord* indirect, VkBuffer argumentBuffer, VkDeviceSize argumentOffset, const DrawTraceContext* trace, std::uint32_t& subdraw, std::optional<std::uint32_t> pieceFirstIndex = std::nullopt) {
    if (recordDrawCommandsImpl(context, commands, state, draw, inputs, indirect, argumentBuffer, argumentOffset, trace, subdraw, pieceFirstIndex)) NoteGuestWorkIn(context, commands);
}

// The GPU-side records of an indirect draw, recorded outside the render pass: the stores that
// produced them (a dispatch in place, a fill, the host) precede the indirect read, as for
// DISPATCH_INDIRECT; a Constant dimension is rewritten in a scratch copy (vkCmdDrawIndirect reads
// the record's dword as the first vertex / instance, and the constant is what the fixed-function
// fetch must start at). The scratch is kept until the batch completed. Returns whether a copy was
// rewritten; `argumentBuffer`/`argumentOffset` receive where the draw reads the records.
void noteIndirectArguments(Recorder* recorder, bool recorded, const IndirectRecord& indirect, VkBuffer& argumentBuffer, VkDeviceSize& argumentOffset) {
    const auto* args = indirect.args;
    if (recorded) {
        // Read in place from the import when the batch runs (a synchronous draw waits for its own).
        recorder->NotePendingRead(args->arguments, static_cast<std::size_t>(args->RangeBytes()), Recorder::ReadKind::Indirect);
        if (args->countIndirect) recorder->NotePendingRead(args->countAddress, 4, Recorder::ReadKind::Indirect);
    }
    argumentBuffer = indirect.argumentImport->buffer;
    argumentOffset = HostImportOffset(*indirect.argumentImport, args->arguments);
}

bool recordIndirectArguments(const Context& context, VkCommandBuffer commands, Recorder* recorder, bool recorded, const IndirectRecord& indirect, std::unique_ptr<DeviceBuffer>& scratch, std::shared_ptr<Buffer>& snapshot, VkBuffer& argumentBuffer, VkDeviceSize& argumentOffset, const std::function<void(std::uint32_t)>& countBarrier, bool snapshots = true) {
    using Rule = Pm4::DrawParameters::IndirectDraw::Rule;
    const auto* args = indirect.args;
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    countBarrier(1);
    const bool rewritten = args->vertexRule == Rule::Constant || args->instanceRule == Rule::Constant;
    const auto bytes = static_cast<std::size_t>(args->RangeBytes());
    const auto* source = reinterpret_cast<const std::byte*>(args->arguments);
    if (snapshots && !args->countIndirect && bytes != 0 && !Recorder::SnapshotWriteOverlaps(args->arguments, bytes) && (recorder == nullptr || recorder->PendingLeaseSerial(args->arguments, bytes) == 0) && GuestMemory::Accessible(source, bytes, false)) {
        snapshot = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
        const auto out = snapshot->Bytes();
        std::memcpy(out.data(), source, bytes);
        for (std::uint32_t record = 0; record < args->count; ++record) {
            const auto base = static_cast<std::size_t>(record) * args->stride;
            if (args->vertexRule == Rule::Constant) std::memcpy(out.data() + base + args->VertexDwordOffset(), &args->vertexConstant, 4);
            if (args->instanceRule == Rule::Constant) std::memcpy(out.data() + base + args->InstanceDwordOffset(), &args->instanceConstant, 4);
        }
        argumentBuffer = snapshot->Handle();
        argumentOffset = 0;
        return rewritten;
    }
    noteIndirectArguments(recorder, recorded, indirect, argumentBuffer, argumentOffset);
    if (!rewritten) return false;
    scratch = std::make_unique<DeviceBuffer>(context, bytes, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    CopyBuffer(context, commands, argumentBuffer, argumentOffset, scratch->Handle(), 0, bytes);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    countBarrier(1);
    const auto update = context.Resolved(&DeviceFunctions::cmdUpdateBuffer, "vkCmdUpdateBuffer");
    for (std::uint32_t record = 0; record < args->count; ++record) {
        const VkDeviceSize base = static_cast<VkDeviceSize>(record) * args->stride;
        if (args->vertexRule == Rule::Constant) update(commands, scratch->Handle(), base + args->VertexDwordOffset(), 4, &args->vertexConstant);
        if (args->instanceRule == Rule::Constant) update(commands, scratch->Handle(), base + args->InstanceDwordOffset(), 4, &args->instanceConstant);
    }
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    countBarrier(1);
    argumentBuffer = scratch->Handle();
    argumentOffset = 0;
    return true;
}

void recordCaptureBarrier(const Context& context, VkCommandBuffer commands) {
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
}

std::shared_ptr<Buffer> meshArgumentBuffer(const Context& context, std::span<const std::uint32_t> restartTable) {
    const auto tableBytes = restartTable.empty() ? 0u : sizeof(std::uint32_t) * (restartTable.size() + 1u);
    auto arguments = std::make_shared<Buffer>(context, ShaderRecompiler::MeshArgumentBytes + tableBytes, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    if (!restartTable.empty()) {
        Require(restartTable.size() <= std::numeric_limits<std::uint32_t>::max(), "mesh restart table is too large");
        const auto length = static_cast<std::uint32_t>(restartTable.size());
        auto* bytes = arguments->Bytes().data() + ShaderRecompiler::MeshArgumentBytes;
        std::memcpy(bytes, &length, sizeof(length));
        std::memcpy(bytes + sizeof(length), restartTable.data(), restartTable.size_bytes());
    }
    return arguments;
}

std::shared_ptr<Buffer> directMeshArguments(const Context& context, const Pm4::DrawParameters& draw, std::span<const std::uint32_t> restartTable) {
    auto arguments = meshArgumentBuffer(context, restartTable);
    const MeshArguments resolved{0u, 0u, 0u, draw.indexCount, 0u};
    std::memcpy(arguments->Bytes().data(), &resolved, sizeof(resolved));
    return arguments;
}

std::shared_ptr<Buffer> recordMeshArguments(const Context& context, VkCommandBuffer commands, Recorder* recorder, bool recorded, const State& state, const Pm4::DrawParameters& draw, const IndirectRecord& indirect, const std::function<void(std::uint32_t)>& countBarrier, std::span<const std::uint32_t> restartTable) {
    const auto* args = indirect.args;
    const auto rules = MeshArgumentRulesFor(context, *state.stages.mesh, draw.indexCount);
    auto arguments = meshArgumentBuffer(context, restartTable);
    if (indirect.path == IndirectDrawPath::Gpu && recorder != nullptr && indirect.argumentImport->address != 0) {
        const std::array<std::uint32_t, 7> words{rules.indexCount, rules.inputSize, rules.step, rules.primitivesPerGroup, rules.maxGroups, rules.maxInstances, rules.maxTotal};
        if (recorder->RecordMeshArguments(commands, indirect.argumentImport->address + (args->arguments - indirect.argumentImport->base), arguments->DeviceAddress(), words)) {
            countBarrier(2);
            if (recorded) recorder->NotePendingRead(args->arguments, static_cast<std::size_t>(args->RangeBytes()), Recorder::ReadKind::Indirect);
            return arguments;
        }
    }
    const auto record = !indirect.records.empty() ? indirect.records.front() : Pm4::ReadDrawArguments(*args, 0);
    const auto resolved = ResolveMeshArguments(record, rules);
    std::memcpy(arguments->Bytes().data(), &resolved, sizeof(resolved));
    return arguments;
}

// Debug aid: APS5_CHECK_INDIRECT_ARGS=1 reads the records of a GPU-side draw back on the CPU once
// its batch completed (the GPU has finished writing them by then) and prints the first 16, counting
// records whose Constant dimension holds a value the hardware would have ignored. Empty otherwise.
std::function<void()> indirectRecordCheck(const IndirectRecord* indirect) {
    static const bool checkArguments = std::getenv("APS5_CHECK_INDIRECT_ARGS") != nullptr;
    if (indirect == nullptr || indirect->args == nullptr || indirect->path != IndirectDrawPath::Gpu || !checkArguments) return {};
    return [args = *indirect->args] {
        using Rule = Pm4::DrawParameters::IndirectDraw::Rule;
        static std::atomic<std::uint64_t> printed{0};
        static std::atomic<std::uint64_t> ignoredValues{0};
        static std::atomic<std::uint64_t> checked{0};
        try {
            for (std::uint32_t record = 0; record < args.count; ++record) {
                const auto arguments = Pm4::ReadDrawArguments(args, record);
                const bool ignoredVertex = args.vertexRule == Rule::Constant && (args.recordBytes == 20 ? arguments.vertexOffset : arguments.firstVertexOrIndex) != 0;
                const bool ignoredInstance = args.instanceRule == Rule::Constant && arguments.firstInstance != 0;
                ++checked;
                if (ignoredVertex || ignoredInstance) ++ignoredValues;
                if (printed.fetch_add(1) < 16) std::fprintf(stderr, "[draw] indirect record 0x%llx: count %u instances %u first %u vertexOffset %u startInstance %u (vertex %s, instance %s)\n", static_cast<unsigned long long>(args.arguments + static_cast<std::uint64_t>(record) * args.stride), arguments.count, arguments.instances, arguments.firstVertexOrIndex, arguments.vertexOffset, arguments.firstInstance, args.vertexRule == Rule::Constant ? "const: record value ignored" : "in-place", args.instanceRule == Rule::Constant ? "const: record value ignored" : "in-place");
            }
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[draw] indirect record read-back failed: %s\n", error.what());
        }
        if (checked % 64 == 0) std::fprintf(stderr, "[draw] indirect records checked %llu, with a value in a Constant dimension %llu\n", static_cast<unsigned long long>(checked.load()), static_cast<unsigned long long>(ignoredValues.load()));
    };
}

std::atomic<std::uint32_t> spaceKeepFailures{0};
std::atomic<int> stopAfterResolve{0};

std::shared_ptr<std::shared_ptr<const void>> reserveSpaceKeep(Recorder& recorder, const ShaderResources& resources) {
    if (!resources.SpaceReuse()) return nullptr;
    if (spaceKeepFailures.load(std::memory_order_relaxed) != 0) {
        spaceKeepFailures.fetch_sub(1, std::memory_order_relaxed);
        throw std::bad_alloc();
    }
    auto slot = std::make_shared<std::shared_ptr<const void>>();
    recorder.Keep(slot);
    return slot;
}

// Everything a recorded draw uses lives until its batch completes.
struct Kept {
    std::shared_ptr<ShaderResources> resources;
    std::shared_ptr<Pipeline> pipeline;
    std::shared_ptr<Framebuffer> framebuffer;
    std::shared_ptr<Buffer> indices;
    std::vector<std::shared_ptr<Buffer>> vertexBuffers;
    std::vector<std::shared_ptr<StorageTexture>> targets;
    std::unique_ptr<DeviceBuffer> scratch;
};

// The completion side of a recorded draw: the kept objects, the record check, the lease outcome,
// the GPU write notes, the copied-buffer write-back (listed in DrawCopiedWriters or run as a
// completion action) and the targets marked dirty.
struct RecordKeeping {
    DrawRecordKeep* keep = nullptr;
    bool passNotes = false;
    std::uint32_t* notesSkipped = nullptr;
    bool* keptOnce = nullptr;
    DrawCommitView* view = nullptr;
};

template <typename Target>
bool keepsOnceFor(const DrawRecordKeep* keep, const Recorder& recorder, const void* resources, const void* pipeline, const void* framebuffer, std::span<const Target> targets, bool plain) {
    return plain && keep != nullptr && keep->recorder == recorder.Id() && keep->batch == recorder.OpenSerial() && keep->resources == resources && keep->pipeline == pipeline && keep->framebuffer == framebuffer && std::ranges::equal(keep->targets, targets, [](const void* kept, const Target& target) { return kept == std::to_address(target); });
}

bool keepsOnce(const DrawRecordKeep* keep, const Recorder& recorder, const ShaderResources* resources, const Pipeline* pipeline, const Framebuffer* framebuffer, std::span<const std::shared_ptr<StorageTexture>> targets, bool plain) {
    return keepsOnceFor<std::shared_ptr<StorageTexture>>(keep, recorder, resources, pipeline, framebuffer, targets, plain);
}

void keepRecordedDraw(Recorder& recorder, const std::shared_ptr<ShaderResources>& resources, std::shared_ptr<Pipeline> pipeline, std::shared_ptr<Framebuffer> framebuffer, DrawInputs& inputs, std::vector<std::shared_ptr<StorageTexture>> targets, std::vector<std::pair<std::uint32_t, std::uint32_t>> slices, VkRect2D scissor, std::unique_ptr<DeviceBuffer> scratch, std::function<void()> checkRecords, bool listed, bool completion, const DrawOutcome& outcome, std::shared_ptr<ShaderResources::DrawUse> use, std::shared_ptr<const void>* spaceKeep, const RecordKeeping& keeping = {}) {
    const bool keptOnce = keepsOnce(keeping.keep, recorder, resources.get(), pipeline.get(), framebuffer.get(), targets, !checkRecords && !completion && use == nullptr && spaceKeep == nullptr);
    std::shared_ptr<Kept> kept;
    if (keptOnce) {
        if (inputs.indices != nullptr || !inputs.vertexBuffers.empty() || scratch != nullptr) {
            auto perDraw = std::make_shared<Kept>();
            perDraw->indices = std::move(inputs.indices);
            perDraw->vertexBuffers = std::move(inputs.vertexBuffers);
            perDraw->scratch = std::move(scratch);
            recorder.Keep(std::move(perDraw));
        }
    } else {
        kept = std::make_shared<Kept>();
        kept->resources = resources;
        kept->pipeline = std::move(pipeline);
        kept->framebuffer = std::move(framebuffer);
        kept->indices = std::move(inputs.indices);
        kept->vertexBuffers = std::move(inputs.vertexBuffers);
        kept->scratch = std::move(scratch);
        kept->targets = std::move(targets);
    }
    if (keeping.keptOnce != nullptr) *keeping.keptOnce = keptOnce;
    const auto& residents = keptOnce ? targets : kept->targets;
    if (!keptOnce) {
        resources->KeepStorageImageBytes(recorder);
        for (const auto& target : residents) {
            if (target != nullptr && !target->Cached()) recorder.KeepBytes(target.get(), static_cast<std::size_t>(target->BudgetBytes()));
        }
        recorder.Keep(kept);
        if (resources != nullptr) recorder.KeepBytes(resources.get(), resources->CopiedBytes());
        if (keeping.keep != nullptr) {
            auto& stamp = *keeping.keep;
            stamp.recorder = recorder.Id();
            stamp.batch = recorder.OpenSerial();
            stamp.resources = kept->resources.get();
            stamp.pipeline = kept->pipeline.get();
            stamp.framebuffer = kept->framebuffer.get();
            stamp.targets.clear();
            for (const auto& target : kept->targets) stamp.targets.push_back(target.get());
        }
    }
    if (checkRecords) recorder.OnComplete(std::move(checkRecords));
    // Counted for the [address-sync] line: a lease released by the completion (the pin waiter
    // finishes the recorder up to the open batch, which holds the kept resources and gets the
    // next serial), or waited for by the caller (any wait releases it at once).
    if (resources->HoldsLease()) CountLeaseOutcome(outcome.waited, outcome.waited ? 0 : recorder.Submissions() + 1, !resources->BdaReadOnly());
    // Every use of a (possibly shared) resources object registers its GPU writes anew.
    if (!keptOnce || !resources->RecordLean()) resources->MarkGpuWrites(recorder);
    NoteHtileWriter_nid_no_patch("draw", 0, resources->GpuWrites());
    if (spaceKeep != nullptr) *spaceKeep = resources->ReleaseSpace();
    // The write-back (fault check, copied buffers) runs when the batch completed; a fault is
    // reported by the recorder ("deferred write-back failed") instead of thrown out of the draw.
    if (listed && resources->HasCopiedWrites()) {
        // As VulkanDevice::dispatch lists its copied writers: delisted before the write-back
        // (one that fails must not keep indirect dispatches on the CPU), listed after the
        // registration (a throw there leaves nothing behind).
        auto writers = DrawCopiedWriters();
        recorder.OnComplete([resources, writers] {
            writers->erase(std::remove(writers->begin(), writers->end(), resources), writers->end());
            resources->WriteBackBuffers();
        });
        writers->push_back(resources);
        NoteCopiedWriter(true);
        GuestMemory::BreakProofs(GuestMemory::ProofBreak::Draw);
    } else if (completion) {
        if (use != nullptr) recorder.OnComplete([resources, use] { resources->CompleteDrawUse(*use); });
        else if (resources->Reusable()) {
            if (!resources->FaultBufferLean()) recorder.OnComplete([resources] { resources->CheckFaults(); });
            else if (resources->TakeFaultCheck(recorder.Id(), recorder.OpenSerial())) recorder.OnComplete([resources] { resources->CheckFaults(); });
        }
        else recorder.OnComplete([resources] { resources->WriteBackBuffers(); });
    }
    const bool storeMask = residents.size() <= 64 && RuntimeSwitches::Set(RuntimeSwitch::KeepFastPath);
    std::uint64_t storeBits = 0;
    std::vector<bool> storeNow(storeMask ? 0 : residents.size(), false);
    if (keeping.view != nullptr) keeping.view->notes.assign(residents.size(), DrawCommitNote{});
    for (std::size_t index = 0; index < residents.size(); ++index) {
        if (residents[index] == nullptr) continue;
        auto* note = keeping.view != nullptr ? &keeping.view->notes[index] : nullptr;
        if (note != nullptr) note->present = true;
        if (residents[index]->Descriptor().tileMode == TextureTileMode::kLinear) {
            if (note != nullptr) note->linear = true;
            const bool store = !residents[index]->MarkDrawn(index < slices.size() ? slices[index].first : 0u, index < slices.size() ? slices[index].second : 0u, scissor) || !residents[index]->Cached();
            if (storeMask) storeBits |= static_cast<std::uint64_t>(store) << index;
            else storeNow[index] = store;
        } else {
            const auto slice = index < slices.size() ? slices[index].first : ~0u;
            const auto count = index < slices.size() ? slices[index].second : 0u;
            if (note != nullptr) {
                const auto probe = residents[index]->ProbeRenote(recorder.Id(), recorder.RenderPassGeneration(), slice, count);
                note->renote = probe.renote;
                note->noteFirst = probe.first;
                note->noteLayers = probe.layers;
                note->storesKeys = residents[index]->FirstWriteStoresKeys();
            }
            if (keeping.passNotes && residents[index]->RenoteInPass(recorder.Id(), recorder.RenderPassGeneration(), slice, count)) {
                if (keeping.notesSkipped != nullptr) ++*keeping.notesSkipped;
                continue;
            }
            if (index < slices.size()) residents[index]->MarkSliceDirty(slice, count);
            else residents[index]->MarkDirty();
            if (note != nullptr) std::tie(note->markedFirst, note->markedLayers) = residents[index]->MarkedLayers();
            if (recorder.RenderPassOpen()) residents[index]->RememberPassNote(recorder.Id(), recorder.RenderPassGeneration(), slice, count);
        }
    }
    for (std::size_t index = 0; index < residents.size(); ++index) {
        if (storeMask ? (storeBits >> index & 1u) != 0 : storeNow[index]) StorageTexture::FlushPending(residents[index]->Descriptor().baseAddress, static_cast<std::size_t>(residents[index]->GuestBytes()), nullptr, "linear target");
    }
}

// What recordDraw records from: the recorder, the resources, the pipeline and framebuffer, the
// resident targets with their views, the indirect records (null for a direct draw), how the
// completion work is handled (`listed`, `completion`, `waited`: see Draw) and, from a recipe, the
// push constant block (else assembled from the stages).
struct StagedWrite {
    std::shared_ptr<StagedTemporary> temporary;
    VkImageView target = VK_NULL_HANDLE;
    const HeapSlotOwner* targetOwner = nullptr;
    std::uint16_t mask = 0;
    std::uint32_t samples = 1;
    VkRect2D area{};
};

// What a record bound the last time (APS5_XP_BINDINGS_AS_IS, a ceiling experiment).
struct BindingsAsIs {
    const ShaderResources* resources = nullptr;
    const ShaderResources::PushPlan* push = nullptr;
    std::shared_ptr<ShaderResources::DrawBindings> bindings;
};

struct RecordedDraw {
    Recorder* recorder = nullptr;
    const State* masked = nullptr;
    std::shared_ptr<ShaderResources> resources;
    std::shared_ptr<Pipeline> pipeline;
    std::shared_ptr<Framebuffer> framebuffer;
    std::span<const VkImageView> targetViews;
    std::vector<std::shared_ptr<StorageTexture>> targets;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> targetSlices;
    std::vector<std::shared_ptr<void>> multisampleTargets;
    std::vector<std::shared_ptr<StorageTexture>> proxies;
    std::vector<VkImage> multisampleImages;
    std::vector<StagedWrite> staged;
    const IndirectRecord* indirect = nullptr;
    std::span<const ShaderResources::MovedBuffer> moved;
    std::span<const std::uint32_t> contentKey;
    bool replay = false;
    bool listed = false;
    bool completion = false;
    bool waited = false;
    const std::array<std::byte, PipelinePushConstantBytes>* pushBytes = nullptr;
    VkShaderStageFlags pushStages = 0;
    std::vector<std::pair<std::uint32_t, std::shared_ptr<Pipeline>>> stencilReplay;
    std::vector<SampleOrderTarget> reordered;
    SampleOrderPass reorderPass;
    std::optional<DepthSampleLocations> depthLocations;
    std::shared_ptr<std::shared_ptr<const void>>* spaceKeep = nullptr;
    DrawRecordAccess* access = nullptr;
    RecordKeeping keeping;
    const ShaderResources::PushPlan* push = nullptr;
    std::shared_ptr<Pipeline> pushPipeline;
    std::shared_ptr<Framebuffer> pushFramebuffer;
    bool* pushed = nullptr;
    DrawCommitView* view = nullptr;
    const ShaderResources::PreparedBuffers* prepared = nullptr;
    const ShaderResources::PreparedBuffers* preparedCheck = nullptr;
    DrawRecordBuffers* buffers = nullptr;
    ShaderResources::PreparedDifference* bufferDifference = nullptr;
    BindingsAsIs* bindingsAsIs = nullptr;
    ShaderResources::BindingRoute* route = nullptr;
    std::shared_ptr<Pipeline> dynamicPipeline;
    std::shared_ptr<Framebuffer> dynamicFramebuffer;
    std::shared_ptr<Pipeline> pushDynamicPipeline;
    std::shared_ptr<Framebuffer> pushDynamicFramebuffer;
    DrawRecordBindings* bindings = nullptr;
    DrawFragment* fragment = nullptr;
    bool fragmentVerify = false;
    DrawRecordFragment* fragmentNote = nullptr;
};

// A remapped viewport depth range reads its scale and offset from the draw's push block (the vertex shader's
// RemapPositionDepth), so draws that differ only in them share one pipeline; the draw's own state supplies them.
void patchDepthRemapPush(std::array<std::byte, PipelinePushConstantBytes>& block, const State& state) {
    if (state.depthRemap.mode == 0u) return;
    static_assert(ShaderRecompiler::DepthRemapPushOffsetBytes + ShaderRecompiler::DepthRemapPushBytes <= PipelinePushSlotBytes);
    const std::array<float, ShaderRecompiler::DepthRemapPushBytes / 4> words{state.depthRemap.scale, state.depthRemap.offset};
    std::memcpy(block.data() + ShaderRecompiler::DepthRemapPushOffsetBytes, words.data(), sizeof(words));
}

bool adjustedInPlace(const ShaderResources::MovedBuffer& entry) {
    return entry.inPlace != VK_NULL_HANDLE && entry.bytes == 0;
}

void recordStencilReplay(const Context& context, CommandState& recording, const State& state, const Pm4::DrawParameters& draw, const DrawInputs& inputs, std::span<const std::pair<std::uint32_t, std::shared_ptr<Pipeline>>> replay, std::span<const CompiledShader> shaders, const ShaderResources& resources, const std::array<std::byte, PipelinePushConstantBytes>* bytes = nullptr, VkShaderStageFlags stages = 0, const ShaderResources::DrawBindings* drawBindings = nullptr) {
    const auto commands = recording.Commands();
    State selected = state;
    auto block = bytes != nullptr ? *bytes : AssemblePushConstants(shaders, RuntimeSwitches::Set(RuntimeSwitch::PushMemcpy));
    if (bytes == nullptr) {
        resources.PatchPushConstants(block);
        stages = PushConstantStages(shaders);
    }
    patchDepthRemapPush(block, state);
    resources.PatchDataAddresses(block, shaders, drawBindings);
    const auto fragment = std::ranges::find_if(shaders, [](const CompiledShader& shader) { return shader.stage == ShaderRecompiler::ShaderStage::Fragment && !shader.program->stencilReferences.empty(); });
    Require(fragment != shaders.end(), "stencil replay fragment stage is missing");
    const auto referenceOffset = (fragment->program->dataAddressPush ? 0u : fragment->pushConstantOffset) + fragment->program->stencilReferenceDword * sizeof(std::uint32_t);
    Require(referenceOffset <= block.size() - sizeof(std::uint32_t), "stencil replay reference exceeds the push block");
    const auto run = [&](const Pm4::DrawParameters& piece, std::optional<std::uint32_t> firstIndex) {
        std::uint32_t unusedSubdraw = 0;
        for (const auto& [reference, pipeline] : replay) {
            selected.stencilReplayReference = reference;
            selected.stencilFront.reference = reference;
            selected.stencilBack.reference = reference;
            recording.SetStencilReference(context, VK_STENCIL_FACE_FRONT_AND_BACK, reference);
            std::memcpy(block.data() + referenceOffset, &reference, sizeof(reference));
            pipeline->PushConstants(commands, stages, block);
            recordDrawCommands(context, commands, selected, piece, inputs, nullptr, VK_NULL_HANDLE, 0, nullptr, unusedSubdraw, firstIndex);
        }
    };
    if (replay.size() == 1u) {
        run(draw, std::nullopt);
        return;
    }
    const auto indexCount = draw.indexed ? inputs.indexCount.value_or(draw.indexCount) : draw.indexCount;
    for (std::uint32_t instance = 0; instance < draw.instanceCount; ++instance) {
        for (std::uint32_t first = 0; first < indexCount - indexCount % 3u; first += 3u) {
            auto piece = draw;
            piece.instanceCount = 1u;
            piece.firstInstance = draw.firstInstance + instance;
            piece.indexCount = 3u;
            if (!draw.indexed) piece.firstVertex += first;
            run(piece, draw.indexed ? first : 0u);
        }
    }
}

void pushDrawConstants(const Pipeline& pipeline, VkCommandBuffer commands, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, const ShaderResources& resources, const std::array<std::byte, PipelinePushConstantBytes>* bytes, VkShaderStageFlags stages, VkDeviceAddress meshArguments = 0, bool restartTable = false, std::uint32_t meshLayerStride = 0, const ShaderResources::DrawBindings* drawBindings = nullptr) {
    auto block = bytes != nullptr ? *bytes : AssemblePushConstants(shaders, RuntimeSwitches::Set(RuntimeSwitch::PushMemcpy));
    if (bytes == nullptr) {
        resources.PatchPushConstants(block);
        stages = PushConstantStages(shaders);
    }
    patchDepthRemapPush(block, state);
    resources.PatchDataAddresses(block, shaders, drawBindings);
    if (state.stages.tessellation) {
        for (const auto& shader : shaders) Require(shader.program->pushConstants.empty() || shader.pushConstantOffset >= PipelinePushSlotBytes || shader.pushConstantOffset + shader.program->pushConstants.size() <= ShaderRecompiler::TessellationLevelPushOffsetBytes, "a stage's push constants overlap the tessellation levels");
        const std::array<float, ShaderRecompiler::TessellationLevelPushBytes / 4> levels{state.stages.minTessellationLevel, state.stages.maxTessellationLevel};
        static_assert(ShaderRecompiler::TessellationLevelPushOffsetBytes + ShaderRecompiler::TessellationLevelPushBytes == PipelinePushSlotBytes);
        std::memcpy(block.data() + ShaderRecompiler::TessellationLevelPushOffsetBytes, levels.data(), sizeof(levels));
    }
    const auto capturedItems = state.tessellationCapture != nullptr ? state.tessellationCapture->items->DeviceAddress() : 0;
    if (!state.stages.mesh) {
        if (state.tessellationCapture != nullptr) {
            for (const auto& shader : shaders) Require(shader.program->pushConstants.empty() || shader.pushConstantOffset >= PipelinePushSlotBytes || shader.pushConstantOffset + shader.program->pushConstants.size() <= ShaderRecompiler::MeshDrawPushOffsetBytes, "a stage's push constants overlap the tessellation capture addresses");
            const auto header = state.tessellationCapture->header->DeviceAddress();
            const std::array<std::uint32_t, 4> words{static_cast<std::uint32_t>(capturedItems), static_cast<std::uint32_t>(capturedItems >> 32u), static_cast<std::uint32_t>(header), static_cast<std::uint32_t>(header >> 32u)};
            static_assert(ShaderRecompiler::MeshDrawPushOffsetBytes + sizeof(std::uint32_t) * 4 <= ShaderRecompiler::TessellationLevelPushOffsetBytes);
            std::memcpy(block.data() + ShaderRecompiler::MeshDrawPushOffsetBytes, words.data(), sizeof(words));
            stages |= VK_SHADER_STAGE_GEOMETRY_BIT;
        }
        pipeline.PushConstants(commands, stages, block);
        return;
    }
    const auto firstVertex = draw.indirect ? draw.indirect->vertexConstant : draw.firstVertex;
    const auto firstInstance = draw.indirect ? draw.indirect->instanceConstant : draw.firstInstance;
    std::array<std::uint32_t, ShaderRecompiler::MeshDrawPushBytes / 4> words{draw.indexCount, firstVertex, firstInstance, (draw.indexed ? draw.indexSize | (restartTable ? ShaderRecompiler::MeshIndexRestartTable : 0u) : 0u) | meshLayerStride << ShaderRecompiler::MeshLayerStrideShift, static_cast<std::uint32_t>(meshArguments), static_cast<std::uint32_t>(meshArguments >> 32u)};
    if (state.tessellationCapture != nullptr) words = {static_cast<std::uint32_t>(capturedItems), static_cast<std::uint32_t>(capturedItems >> 32u), 0u, 0u, static_cast<std::uint32_t>(meshArguments), static_cast<std::uint32_t>(meshArguments >> 32u)};
    static_assert(ShaderRecompiler::MeshDrawPushOffsetBytes + ShaderRecompiler::MeshDrawPushBytes == PipelinePushSlotBytes);
    std::memcpy(block.data() + ShaderRecompiler::MeshDrawPushOffsetBytes, words.data(), sizeof(words));
    pipeline.PushConstants(commands, stages | VK_SHADER_STAGE_MESH_BIT_EXT, block);
}

// The record of a draw whose targets are all resident (`lean`: recorded into the device's open
// batch, rendering in the general layout): one memory barrier before the pass (earlier writes to
// the attachments and the draw's inputs visible to it) and one after it, recorded by the recorder
// when the pass ends (Recorder::LeaveRenderPassOpen), so consecutive draws into the same
// attachments with nothing recorded between them share one pass; the host-write and host-read
// barriers are implicit in the submission and the fence. Then the kept objects, the completion
// work and, for a waited-for draw, the recorder sync.
bool CaptureInputsEnabled() {
    static const bool enabled = std::getenv("APS5_CAPTURE_INPUTS") != nullptr;
    Require(!enabled || CaptureTrace::Enabled(), "APS5_CAPTURE_INPUTS requires APS5_CAPTURE_TRACE");
    return enabled;
}

void captureInputs(const Context& context, Recorder& recorder, VkCommandBuffer commands, const ShaderResources& resources, const std::shared_ptr<ShaderResources::DrawBindings>& bindings, std::uint64_t target) {
    struct Sample {
        std::uint64_t address;
        std::size_t offset;
        std::vector<std::byte> expected;
        VkBuffer source;
        VkDeviceSize sourceOffset;
    };
    static unsigned long long nextDraw = 0;
    const auto draw = ++nextDraw;
    const auto batch = static_cast<unsigned long long>(recorder.Submissions() + 1);
    std::vector<Sample> samples;
    std::size_t total = 0;
    const auto addSample = [&](std::uint64_t address, std::size_t bytes, VkBuffer source, VkDeviceSize offset, const std::byte* expected) {
        if (bytes == 0 || bytes > 512) return;
        Require(offset % 4 == 0 && bytes % 4 == 0, "capture input is not aligned for a Vulkan buffer copy");
        Require(total + bytes <= 65536, "capture inputs exceed 64 KiB per draw");
        Sample sample{address, total, std::vector<std::byte>(bytes), source, offset};
        std::memcpy(sample.expected.data(), expected, bytes);
        total += bytes;
        samples.push_back(std::move(sample));
    };
    if (bindings != nullptr) {
        for (const auto& snapshot : bindings->snapshots) {
            if (snapshot.buffer == nullptr) continue;
            const auto bytes = snapshot.Bytes();
            addSample(snapshot.address, bytes.size(), snapshot.buffer->Handle(), snapshot.offset, bytes.data());
        }
    }
    for (const auto& [begin, end] : resources.InPlaceReads()) {
        Require(end >= begin, "invalid capture input range");
        const auto bytes = static_cast<std::size_t>(end - begin);
        if (bytes == 0 || bytes > 512) continue;
        if (bindings != nullptr && std::any_of(bindings->snapshots.begin(), bindings->snapshots.end(), [&](const auto& snapshot) { return snapshot.address < end && begin < snapshot.address + snapshot.bytes; })) continue;
        if (resources.WritesOverlap(begin, bytes) || recorder.PendingWriteOverlaps(begin, bytes)) {
            CaptureTrace::Log("input-skip draw=%llu batch=%llu address=%llx bytes=%zu reason=gpu-writer", draw, batch, static_cast<unsigned long long>(begin), bytes);
            continue;
        }
        const auto* imported = HostImportFor(context, begin, bytes);
        Require(imported != nullptr, "capture input has no host import");
        addSample(begin, bytes, imported->buffer, HostImportOffset(*imported, begin), reinterpret_cast<const std::byte*>(begin));
    }
    CaptureTrace::Log("input-capture draw=%llu batch=%llu target=%llx ranges=%zu bytes=%zu", draw, batch, static_cast<unsigned long long>(target), samples.size(), total);
    if (samples.empty()) return;
    auto readback = std::make_shared<Buffer>(context, total, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    const auto copy = context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
    for (const auto& sample : samples) {
        const VkBufferCopy region{sample.sourceOffset, sample.offset, sample.expected.size()};
        copy(commands, sample.source, readback->Handle(), 1, &region);
    }
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
    recorder.OnComplete([draw, batch, target, readback, samples = std::move(samples)] {
        const auto actual = readback->Bytes();
        std::size_t differences = 0;
        for (const auto& sample : samples) {
            std::size_t changed = 0;
            for (std::size_t index = 0; index < sample.expected.size(); ++index) {
                const auto gpu = actual[sample.offset + index];
                if (sample.expected[index] == gpu) continue;
                if (changed < 16) CaptureTrace::Log("input-byte draw=%llu batch=%llu address=%llx offset=%zu cpu=%02x gpu=%02x", draw, batch, static_cast<unsigned long long>(sample.address), index, std::to_integer<unsigned>(sample.expected[index]), std::to_integer<unsigned>(gpu));
                ++changed;
            }
            if (changed != 0) {
                ++differences;
                CaptureTrace::Log("input-mismatch draw=%llu batch=%llu target=%llx address=%llx bytes=%zu changed=%zu", draw, batch, static_cast<unsigned long long>(target), static_cast<unsigned long long>(sample.address), sample.expected.size(), changed);
            }
        }
        CaptureTrace::Log("input-result draw=%llu batch=%llu ranges=%zu mismatched=%zu", draw, batch, samples.size(), differences);
    });
}

bool WriteOnlyPasses() {
    static const bool enabled = std::getenv("APS5_NO_WRITE_ONLY_PASSES") == nullptr;
    return enabled;
}

// The template's own part of a draw's pass access (APS5_NO_PASS_ACCESS_MEMO turns it off): its
// device reads after the write-only filter, its hazard writes and write-only ranges, its storage
// images and whether it reads one of its targets. The same template drawn into the same targets
// within the same render pass takes them from here instead of deriving them again; the draw's own
// reads (in-place vertex and index ranges, GPU-side indirect records) are added per draw. The
// pass keeps the template alive, so the pointer cannot name another object while it lasts.
// Address-based and address-table builds never use it.
using PassAccessMemo = DrawRecordAccess;
struct PassAccessMemoStorage {};

PassAccessMemo& passAccessMemo() {
    return HostThreadLocal<PassAccessMemo, PassAccessMemoStorage>();
}

// APS5_VERIFY_PASS_ACCESS_MEMO: every draw the memo would serve derives its access again and records
// with it; the ones whose kept access differs are counted ([pass-access-verify] on stderr).
bool PassAccessMemoVerify() {
    static const bool on = std::getenv("APS5_VERIFY_PASS_ACCESS_MEMO") != nullptr;
    return on;
}

template <typename Target>
bool passAccessMatchesFor(const DrawRecordAccess& memo, const Recorder* recorder, const ShaderResources& resources, std::span<const Target> targets) {
    return memo.recorder == recorder && memo.generation == recorder->RenderPassGeneration() && memo.resources == &resources && std::ranges::equal(memo.targets, targets, [](const void* kept, const Target& target) { return kept == std::to_address(target); });
}

bool passAccessMatches(const DrawRecordAccess& memo, const Recorder* recorder, const ShaderResources& resources, std::span<const std::shared_ptr<StorageTexture>> targets) {
    return passAccessMatchesFor<std::shared_ptr<StorageTexture>>(memo, recorder, resources, targets);
}

void notePassAccessCheck(bool differed) {
    static std::atomic<std::uint64_t> checks{0}, differences{0};
    if (differed) differences.fetch_add(1, std::memory_order_relaxed);
    const auto count = checks.fetch_add(1, std::memory_order_relaxed) + 1u;
    if (differed || count % 262144u == 0u || (count & (count - 1u)) == 0u) std::fprintf(stderr, "[pass-access-verify] %llu checked, %llu differed\n", static_cast<unsigned long long>(count), static_cast<unsigned long long>(differences.load()));
}

void noteReplayPassCensus(const Recorder* recorder, const ShaderResources* resources, const Pipeline* pipeline, const Framebuffer* framebuffer, bool continued, bool counted) {
    thread_local std::array<const void*, 4> last{};
    const std::array<const void*, 4> now{recorder, resources, pipeline, framebuffer};
    if (counted) NoteReplayPass(continued, last[0] == now[0] && last[1] == now[1], last[2] == now[2] && last[3] == now[3]);
    last = now;
}

bool sameFragmentBound(const ShaderResources::BoundBuffer& left, const ShaderResources::BoundBuffer& right) {
    if (left.set != right.set || left.binding != right.binding || left.element != right.element || left.snapshot != right.snapshot || left.info.range != right.info.range) return false;
    if (!left.snapshot) return left.info.buffer == right.info.buffer && left.info.offset == right.info.offset;
    return left.address == right.address && left.bytes == right.bytes && left.content == right.content;
}

void keepFragmentBindings(Recorder& recorder, const std::shared_ptr<ShaderResources::DrawBindings>& bindings) {
    recorder.Keep(bindings);
    if (RuntimeSwitches::Set(RuntimeSwitch::KeepBytesOnce)) {
        for (const auto& snapshot : bindings->snapshots) {
            if (snapshot.buffer != nullptr) recorder.KeepSnapshotBytes(snapshot.buffer.get(), snapshot.buffer->Bytes().size());
        }
        return;
    }
    for (const auto& snapshot : bindings->snapshots) {
        if (snapshot.buffer != nullptr) recorder.KeepBytes(snapshot.buffer.get(), snapshot.buffer->Bytes().size());
    }
}

void keepFragmentBindingsMoved(Recorder& recorder, std::shared_ptr<ShaderResources::DrawBindings>& bindings) {
    const auto* kept = bindings.get();
    recorder.Keep(std::move(bindings));
    if (RuntimeSwitches::Set(RuntimeSwitch::KeepBytesOnce)) {
        for (const auto& snapshot : kept->snapshots) {
            if (snapshot.buffer != nullptr) recorder.KeepSnapshotBytes(snapshot.buffer.get(), snapshot.buffer->Bytes().size());
        }
        return;
    }
    for (const auto& snapshot : kept->snapshots) {
        if (snapshot.buffer != nullptr) recorder.KeepBytes(snapshot.buffer.get(), snapshot.buffer->Bytes().size());
    }
}

void keepExpressBindings(Recorder& recorder, std::shared_ptr<ShaderResources::DrawBindings>& bindings) {
    if (RuntimeSwitches::Set(RuntimeSwitch::KeepBindingsMoved)) keepFragmentBindingsMoved(recorder, bindings);
    else keepFragmentBindings(recorder, bindings);
}

namespace {

bool samplesImage(const ShaderResources& resources, VkImage image) {
    if (image == VK_NULL_HANDLE) return false;
    const auto& textures = resources.SampledTextures();
    return std::any_of(textures.begin(), textures.end(), [&](const std::shared_ptr<Texture>& texture) { return texture != nullptr && texture->Image() == image; });
}

bool samplesAttachedDepth(const Recorder& recorder, const ShaderResources& resources) {
    return samplesImage(resources, recorder.AttachedDepthImage());
}

bool beginDepthLayout(const Context& context, Recorder& recorder, VkCommandBuffer commands, const Pipeline& pipeline, const Framebuffer& framebuffer, const State& state, const ShaderResources& resources) {
    if (!RuntimeSwitches::Set(RuntimeSwitch::XpDepthLayouts) || !pipeline.DynamicDepth() || !state.depth.has_value() || state.depth->samples != 1u || state.rasterizerDiscard || state.customSampleLocations || GpuPathEnabled()) return false;
    const auto views = framebuffer.Views();
    if (views.empty()) return false;
    const auto attached = DepthSurfaceAttachmentImage(context, views.back());
    if (attached.image == VK_NULL_HANDLE || samplesImage(resources, attached.image)) return false;
    return recorder.BeginDepthLayout(commands, attached.image, attached.aspects);
}

VkImage passRestartDepth(const Context& context, const Framebuffer& framebuffer, const State& state, const ShaderResources& resources) {
    if (!RuntimeSwitches::Set(RuntimeSwitch::XpTargetRestarts) || !state.depth.has_value() || state.depth->samples != 1u || GpuPathEnabled()) return VK_NULL_HANDLE;
    const auto views = framebuffer.Views();
    if (views.empty()) return VK_NULL_HANDLE;
    const auto attached = DepthSurfaceAttachmentImage(context, views.back()).image;
    return samplesImage(resources, attached) ? VK_NULL_HANDLE : attached;
}

}

void recordDraw(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, DrawInputs& inputs, RecordedDraw& record, DrawOutcome& outcome, DrawTimer& timer, double& ownWaitedMs, const DrawTraceContext* trace = nullptr) {
    std::uint32_t subdraw = 0;
    auto* recorder = record.recorder;
    auto& resources = *record.resources;
    const auto* args = record.indirect != nullptr ? record.indirect->args : nullptr;
    const bool gpuIndirect = args != nullptr && record.indirect->path == IndirectDrawPath::Gpu;
    using CommandClass = Recorder::CommandClass;
    const auto countBarrier = [&](std::uint32_t count) { Recorder::CountBarriers(CommandClass::Draw, count); };
    // The pass is named by its attachment views (stable while the kept targets live, so unique
    // within the open batch) and the extent. The previous draw's pass is continued only when this
    // draw neither reads its attachments (a barrier would be owed, which no pass allows) nor
    // records anything outside a pass (an indirect draw's argument barrier and scratch copies).
    const auto passKey = DrawRenderPassKey(context, state, record.targetViews);
    auto& memo = record.access != nullptr ? *record.access : passAccessMemo();
    const bool memoUsable = !resources.HoldsLease() && !resources.UsesAddressTable() && (record.access != nullptr || !RuntimeSwitches::Set(RuntimeSwitch::NoPassAccessMemo));
    const bool memoMatched = memoUsable && passAccessMatches(memo, recorder, resources, record.targets);
    const bool memoHit = memoMatched && !PassAccessMemoVerify() && record.view == nullptr;
    const bool readsTarget = memoHit ? memo.readsTarget : std::any_of(record.targets.begin(), record.targets.end(), [&](const std::shared_ptr<StorageTexture>& target) { return resources.ReadsImage(target.get()); });
    const bool movedInPlace = std::ranges::any_of(record.moved, [](const ShaderResources::MovedBuffer& entry) { return entry.inPlace != VK_NULL_HANDLE; });
    const bool movedBehind = std::ranges::any_of(record.moved, [](const ShaderResources::MovedBuffer& entry) { return entry.inPlace != VK_NULL_HANDLE && entry.behind; });
    // A queued DCC key store over memory the draw writes or reads in place (unknown for an
    // address-based build), or over its GPU-side records, lands before it, as before a dispatch
    // (VulkanDevice::dispatch); the flush ends an open pass, so it comes before the decision.
    const auto touches = [&](std::uint64_t begin, std::uint64_t end) {
        const auto bytes = static_cast<std::size_t>(end - begin);
        if (resources.WritesOverlap(begin, bytes) || resources.ReadsOverlap(begin, bytes)) return true;
        if (movedInPlace && std::ranges::any_of(record.moved, [&](const ShaderResources::MovedBuffer& entry) { return entry.inPlace != VK_NULL_HANDLE && begin < entry.begin + entry.bytes && entry.begin < end; })) return true;
        if (!gpuIndirect) return false;
        return (begin < args->arguments + args->RangeBytes() && args->arguments < end) || (args->countIndirect && begin < args->countAddress + 4 && args->countAddress < end);
    };
    if (recorder->HasQueuedKeyStores() && (resources.HoldsLease() || recorder->AnyQueuedKeyStore(touches))) {
        recorder->FlushKeyStores();
        outcome.keyStoresFlushed = true;
    }
    // A queued label store over such memory likewise (Recorder::RecordStore).
    if (recorder->HasQueuedStores() && (resources.HoldsLease() || recorder->AnyQueuedStore(touches))) {
        recorder->FlushStores();
        outcome.storesFlushed = true;
    }
    const bool capture = CaptureInputsEnabled();
    const bool meshIndirect = state.stages.mesh && args != nullptr;
    const bool captured = state.stages.mesh && state.tessellationCapture != nullptr;
    const bool addressBased = resources.HoldsLease();
    struct PassReadsStorage {};
    auto& passReads = HostThreadLocal<std::vector<std::pair<std::uint64_t, std::uint64_t>>, PassReadsStorage>();
    passReads.clear();
    if (memoHit) passReads.assign(memo.reads.begin(), memo.reads.end());
    else if (!addressBased) resources.DeviceReads(passReads);
    const auto deviceReads = passReads.size();
    if (gpuIndirect) {
        passReads.emplace_back(args->arguments, args->arguments + args->RangeBytes());
        if (args->countIndirect) passReads.emplace_back(args->countAddress, args->countAddress + 4);
    }
    passReads.insert(passReads.end(), inputs.inPlaceReads.begin(), inputs.inPlaceReads.end());
    for (const auto& entry : record.moved) {
        if (entry.inPlace == VK_NULL_HANDLE) continue;
        const auto begin = entry.bytes != 0 ? entry.begin : entry.address - entry.adjustment;
        passReads.emplace_back(begin, begin + (entry.bytes != 0 ? entry.bytes : GuestBufferMemory::ViewBytes(entry.size, entry.adjustment)));
    }
    std::vector<std::pair<VkImage, bool>> ownImages;
    if (!memoHit) ownImages = resources.StorageImages();
    const std::span<const std::pair<VkImage, bool>> passImages = memoHit ? std::span<const std::pair<VkImage, bool>>(memo.images) : std::span<const std::pair<VkImage, bool>>(ownImages);
    std::vector<std::pair<VkImage, bool>> passAttachments;
    passAttachments.reserve(record.targets.size());
    for (const auto& target : record.targets) {
        if (target != nullptr) passAttachments.emplace_back(target->Image(), true);
    }
    for (const auto image : record.multisampleImages) passAttachments.emplace_back(image, true);
    const bool unknownMemory = addressBased || resources.UsesAddressTable() || resources.HasCopiedWrites();
    static_cast<void>(StorageTexture::StoreDrawnLinearTouching("draw access", [&](std::uint64_t begin, std::uint64_t end) { return std::any_of(passReads.begin(), passReads.end(), [&](const auto& read) { return read.first < end && begin < read.second; }) || resources.WritesOverlap(begin, static_cast<std::size_t>(end - begin)); }, unknownMemory));
    const auto drawUse = resources.BeginDrawUse(*recorder);
    ShaderResources::HeapReplayProof replayProof;
    const bool hitCensus = ReplayCensusEnabled() && outcome.kind == KindTemplateHit && resources.HeapStage();
    const auto* faultRecord = drawUse != nullptr ? drawUse->record.get() : nullptr;
    std::vector<std::size_t> viewSelected;
    auto* selectedOut = record.view != nullptr || record.preparedCheck != nullptr ? &viewSelected : nullptr;
    using PreparedSelection = ShaderResources::PreparedBuffers::Selection;
    auto preparedSelection = PreparedSelection::Hazard;
    std::shared_ptr<ShaderResources::DrawBindings> keptBindings;
    const bool bindingsKept = record.bindingsAsIs != nullptr && faultRecord == nullptr && !record.replay && record.view == nullptr && record.preparedCheck == nullptr;
    if (bindingsKept && record.bindingsAsIs->resources == &resources && record.bindingsAsIs->push == record.push) keptBindings = record.bindingsAsIs->bindings;
    auto* route = keptBindings == nullptr && !record.replay && record.prepared != nullptr ? record.route : nullptr;
    auto* fragment = record.fragment;
    const auto dropFragment = [&](DrawFragmentDrop why) {
        if (record.fragmentNote != nullptr) record.fragmentNote->drop = why;
        fragment = nullptr;
    };
    if (fragment != nullptr && (faultRecord != nullptr || record.replay || keptBindings != nullptr || record.view != nullptr || record.preparedCheck != nullptr || record.prepared == nullptr || hitCensus || capture)) dropFragment(DrawFragmentDrop::Modes);
    std::shared_ptr<ShaderResources::DrawBindings> fragmentBindings;
    if (fragment != nullptr && !record.fragmentVerify) {
        auto held = resources.FragmentSelectionHolds(*recorder, record.moved, *record.prepared, route);
        if (held == PreparedSelection::Taken && resources.ChecksFaults() && fragment->faultGeneration != resources.FaultGeneration() && fragment->bindings != nullptr) resources.RewriteFragmentFaults(*fragment->bindings);
        if (held == PreparedSelection::Taken) {
            fragmentBindings = fragment->bindings;
            preparedSelection = held;
            if (route != nullptr) {
                route->taken = ShaderResources::BindingTaken::Placed;
                route->bytes = record.prepared->blob.size();
            }
        } else {
            preparedSelection = held;
            dropFragment(DrawFragmentDrop::Selection);
        }
    }
    auto* bindingsCheck = fragment == nullptr && route != nullptr && record.bindings != nullptr && record.bindings->verify ? record.bindings : nullptr;
    std::vector<std::size_t> routeSelected;
    if (bindingsCheck != nullptr && selectedOut == nullptr) selectedOut = &routeSelected;
    const bool bindingsTimed = keptBindings == nullptr && record.bindings != nullptr && record.bindings->timed;
    const auto bindingsBegan = bindingsTimed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto drawBindings = fragmentBindings != nullptr ? fragmentBindings : keptBindings != nullptr ? keptBindings : record.push != nullptr && faultRecord == nullptr ? resources.PreparePushedDrawBindings(*recorder, record.moved, *record.push, selectedOut, record.prepared, &preparedSelection, route) : record.replay ? resources.PrepareReplayDrawBindings(*recorder, record.moved, shaders, faultRecord, replayProof, hitCensus, record.contentKey) : resources.PrepareDrawBindings(*recorder, record.moved, false, faultRecord, hitCensus, selectedOut, record.prepared, &preparedSelection, route);
    timer.lap(FullDrawStep::Bindings);
    if (fragmentBindings != nullptr) keepFragmentBindings(*recorder, fragmentBindings);
    if (fragment != nullptr && record.fragmentVerify && preparedSelection != PreparedSelection::Taken) dropFragment(DrawFragmentDrop::Selection);
    if (bindingsTimed) record.bindings->nanoseconds = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - bindingsBegan).count());
    if (bindingsTimed || route != nullptr || (record.bindings != nullptr && record.bindings->dynamicLeave != 0)) {
        auto& noted = *record.bindings;
        noted.written = drawBindings != nullptr && (drawBindings->allocation.set != VK_NULL_HANDLE || drawBindings->fragmentAllocation.set != VK_NULL_HANDLE);
        noted.pushed = drawBindings != nullptr && drawBindings->pushedSet >= 0;
    }
    if (route != nullptr && record.bindings != nullptr) {
        auto& noted = *record.bindings;
        noted.taken = route->taken;
        noted.tested = route->tested;
        noted.stamped = route->stamped;
        noted.refuted = route->refuted;
        noted.bytes = route->bytes;
        noted.setsLent = route->setsLent;
        noted.setsWritten = route->setsWritten;
        noted.guards = route->guards;
        noted.dynamicBound = route->dynamicBound;
        noted.dynamicHits = route->dynamicHits;
        noted.dynamicWritten = route->dynamicWritten;
        noted.dynamicRefused = route->dynamicRefused;
        noted.dynamicLeave = static_cast<std::uint8_t>(route->leave);
        if (route->taken == ShaderResources::BindingTaken::Placed && drawBindings != nullptr) noted.objects = 1u + (drawBindings->snapshots.empty() ? 0u : 1u) + (drawBindings->pushedSet >= 0 && !drawBindings->pushSelected.empty() ? 1u : 0u);
    }
    if (bindingsCheck != nullptr && (route->taken == ShaderResources::BindingTaken::Placed || route->setsLent != 0 || route->dynamicBound != 0)) {
        std::vector<std::size_t> todaySelected;
        auto todaySelection = PreparedSelection::Hazard;
        const auto todays = record.push != nullptr && faultRecord == nullptr ? resources.PreparePushedDrawBindings(*recorder, record.moved, *record.push, &todaySelected, record.prepared, &todaySelection) : resources.PrepareDrawBindings(*recorder, record.moved, false, faultRecord, false, &todaySelected, record.prepared, &todaySelection);
        if (drawBindings != nullptr && drawBindings->dynamic) resources.DescribeDynamicBindings(*drawBindings, *selectedOut, record.push, bindingsCheck->mine);
        else resources.DescribeDrawBindings(drawBindings.get(), *selectedOut, record.push, bindingsCheck->mine, true);
        resources.DescribeDrawBindings(todays.get(), todaySelected, record.push, bindingsCheck->today, true);
        bindingsCheck->checked = true;
        bindingsCheck->todayTaken = todaySelection == PreparedSelection::Taken;
    }
    if (keptBindings != nullptr) preparedSelection = PreparedSelection::Taken;
    else if (bindingsKept) *record.bindingsAsIs = {&resources, record.push, drawBindings};
    if (record.prepared != nullptr && record.buffers != nullptr) *record.buffers = preparedSelection == PreparedSelection::Taken ? DrawRecordBuffers::Used : preparedSelection == PreparedSelection::DepthGuards ? DrawRecordBuffers::DepthGuards : DrawRecordBuffers::UnmovedHazard;
    if (record.preparedCheck != nullptr && record.buffers != nullptr && record.bufferDifference != nullptr) {
        auto& difference = *record.bufferDifference;
        difference = ShaderResources::ComparePreparedSelection(*record.preparedCheck, drawBindings.get(), viewSelected);
        if (difference.what == nullptr || difference.memoryChanged || difference.unwatched) {
            if (auto bound = resources.ComparePreparedBinding(*recorder, record.moved, *record.preparedCheck, drawBindings.get(), viewSelected); bound.what != nullptr) difference = std::move(bound);
        }
        *record.buffers = difference.what == nullptr ? DrawRecordBuffers::Checked : difference.memoryChanged ? DrawRecordBuffers::MemoryChanged : difference.unwatched ? DrawRecordBuffers::Unwatched : DrawRecordBuffers::Differs;
    }
    if (record.view != nullptr && faultRecord == nullptr && !record.replay && !resources.HeapStage()) {
        resources.DescribeDrawBindings(drawBindings.get(), viewSelected, record.push, record.view->bindings);
        record.view->bound = true;
    }
    const bool pushing = drawBindings != nullptr && drawBindings->pushedSet >= 0;
    if (pushing) {
        record.pipeline = drawBindings->dynamic ? record.pushDynamicPipeline : record.pushPipeline;
        record.framebuffer = drawBindings->dynamic ? record.pushDynamicFramebuffer : record.pushFramebuffer;
        if (record.pushed != nullptr) *record.pushed = true;
    } else if (drawBindings != nullptr && drawBindings->dynamic) {
        record.pipeline = record.dynamicPipeline;
        record.framebuffer = record.dynamicFramebuffer;
    }
    static const bool addressReadPasses = std::getenv("APS5_NO_ADDRESS_READ_PASSES") == nullptr;
    const bool readOnlyBda = addressReadPasses && resources.BdaReadOnly();
    outcome.addressDraw = addressBased;
    const bool unknownWrites = ((addressBased || resources.UsesAddressTable()) && !readOnlyBda) || resources.HasCopiedWrites();
    GuestRangeList passWrites = resources.GpuWrites();
    GuestRangeList passWriteOnly{};
    std::size_t ownReads = deviceReads;
    struct HazardWritesStorage {};
    if (memoHit) {
        passWrites = memo.writes;
        passWriteOnly = memo.writeOnly;
    } else if (const auto& writeOnly = resources.WriteOnlyRanges(); !writeOnly.empty() && !unknownMemory && WriteOnlyPasses()) {
        const auto listed = [&](const std::pair<std::uint64_t, std::uint64_t>& range, bool aligned) {
            return std::any_of(writeOnly.begin(), writeOnly.end(), [&](const auto& only) {
                if (!aligned) return only == range;
                const auto begin = only.first & ~std::uint64_t{3};
                const auto end = begin == only.first ? only.second : (only.second + 3) & ~std::uint64_t{3};
                return range.first == begin && range.second == end;
            });
        };
        const auto ownEnd = passReads.begin() + static_cast<std::ptrdiff_t>(deviceReads);
        const auto kept = std::remove_if(passReads.begin(), ownEnd, [&](const auto& range) { return listed(range, true); });
        ownReads = static_cast<std::size_t>(kept - passReads.begin());
        passReads.erase(kept, ownEnd);
        auto& hazardWrites = HostThreadLocal<std::vector<std::pair<std::uint64_t, std::uint64_t>>, HazardWritesStorage>();
        hazardWrites.clear();
        for (const auto& range : resources.GpuWrites()) {
            if (!listed(range, false)) hazardWrites.push_back(range);
        }
        passWrites = hazardWrites;
        passWriteOnly = writeOnly;
    }
    const PassAccess passAccess{passReads, passWrites, passImages, passAttachments, unknownMemory, unknownWrites, passWriteOnly};
    if (record.view != nullptr && !resources.HoldsLease() && !resources.UsesAddressTable()) {
        auto& derived = record.view->access.emplace();
        derived.readsTarget = readsTarget;
        derived.reads.assign(passReads.begin(), passReads.begin() + static_cast<std::ptrdiff_t>(ownReads));
        derived.writes.assign(passWrites.begin(), passWrites.end());
        derived.writeOnly.assign(passWriteOnly.begin(), passWriteOnly.end());
        derived.images.assign(passImages.begin(), passImages.end());
    }
    if (memoMatched && !memoHit && PassAccessMemoVerify()) {
        const bool same = memo.readsTarget == readsTarget && std::ranges::equal(memo.reads, std::span(passReads).first(ownReads)) && std::ranges::equal(memo.writes, passWrites) && std::ranges::equal(memo.writeOnly, passWriteOnly) && std::ranges::equal(memo.images, passImages);
        notePassAccessCheck(!same);
    }
    const bool fenced = state.depth.has_value() || !record.proxies.empty();
    resources.RequireGuestBuffers();
    if (UsesMeshWindow(shaders)) PrepareMeshWindow(context);
    if (record.replay) resources.ConfirmHeapReplay(*recorder, shaders, drawBindings, replayProof);
    const VkImage restartDepth = record.framebuffer != nullptr && record.proxies.empty() && record.reordered.empty() && !record.depthLocations ? passRestartDepth(context, *record.framebuffer, state, resources) : VK_NULL_HANDLE;
    using IndirectRule = Pm4::DrawParameters::IndirectDraw::Rule;
    const bool passIndirect = RuntimeSwitches::Set(RuntimeSwitch::IndirectInPass);
    const bool indirectForced = gpuIndirect && (!passIndirect || args->vertexRule == IndirectRule::Constant || args->instanceRule == IndirectRule::Constant || !recorder->PassIndirectReads());
    const auto start = recorder->StartDrawPass(passKey, capture || readsTarget || indirectForced || meshIndirect || captured || !record.staged.empty() || PixelFunctionLdsSubgroup(shaders) != 0u || samplesAttachedDepth(*recorder, resources), fenced, passAccess, restartDepth);
    if (ReplayCensusEnabled()) noteReplayPassCensus(recorder, &resources, record.pipeline.get(), record.framebuffer.get(), start.continued, hitCensus);
    // The memo names the pass the draw now records in.
    if (memoUsable && !memoHit) {
        memo.recorder = recorder;
        memo.resources = &resources;
        memo.targets.clear();
        for (const auto& target : record.targets) memo.targets.push_back(target.get());
        memo.readsTarget = readsTarget;
        memo.reads.assign(passReads.begin(), passReads.begin() + static_cast<std::ptrdiff_t>(ownReads));
        memo.writes.assign(passWrites.begin(), passWrites.end());
        memo.writeOnly.assign(passWriteOnly.begin(), passWriteOnly.end());
        memo.images.assign(passImages.begin(), passImages.end());
    }
    if (memoUsable) memo.generation = recorder->RenderPassGeneration();
    if (record.spaceKeep != nullptr) *record.spaceKeep = reserveSpaceKeep(*recorder, resources);
    for (const auto& target : record.multisampleTargets) KeepMultisampleTarget_nid_no_patch(target);
    const bool continued = start.continued;
    outcome.passContinued = continued;
    outcome.passBegun = !continued;
    outcome.passBreak = start.reason;
    const auto commands = start.commands;
    for (const auto& entry : record.moved) {
        if (entry.inPlace != VK_NULL_HANDLE && entry.bytes != 0) recorder->NotePendingRead(entry.begin, entry.bytes, Recorder::ReadKind::DrawInput);
    }
    auto& recording = recorder->GraphicsState();
    if (!continued) {
        recorder->PrepareSampleSlot();
        PerformanceTrace::Amount(PerformanceTrace::RenderPassScope, 1);
    }
    if (capture) captureInputs(context, *recorder, commands, resources, drawBindings, record.targets.empty() || record.targets.front() == nullptr ? 0 : record.targets.front()->Descriptor().baseAddress);
    // The draw's [gputime] class range: from its first barrier to the pass's trailing barrier (a
    // continued draw lies inside its pass's range).
    const auto drawTiming = !continued ? recorder->BeginGpuTiming(CommandClass::Draw) : Recorder::NoTiming;
    if (!continued && Recorder::BarrierValidate()) {
        auto reads = resources.InPlaceReads();
        reads.insert(reads.end(), inputs.inPlaceReads.begin(), inputs.inPlaceReads.end());
        if (gpuIndirect) {
            reads.emplace_back(args->arguments, args->arguments + args->RangeBytes());
            if (args->countIndirect) reads.emplace_back(args->countAddress, args->countAddress + 4);
        }
        for (const auto& entry : record.moved) {
            if (entry.inPlace != VK_NULL_HANDLE) reads.emplace_back(entry.begin, entry.begin + entry.bytes);
        }
        auto images = resources.StorageImages();
        for (const auto& target : record.targets) {
            if (target != nullptr) images.emplace_back(target->Image(), true);
        }
        recorder->NoteAccess(CommandClass::Draw, Recorder::Access{reads, resources.GpuWrites(), images, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | inputs.shaderStages, resources.HoldsLease()});
    }
    std::unique_ptr<DeviceBuffer> scratch;
    std::shared_ptr<Buffer> argumentSnapshot;
    std::shared_ptr<Buffer> meshArguments;
    VkBuffer argumentBuffer = VK_NULL_HANDLE;
    VkDeviceSize argumentOffset = 0;
    bool rewritten = false;
    if (!meshIndirect && !inputs.restartTable.empty()) meshArguments = directMeshArguments(context, draw, inputs.restartTable);
    const auto indirectFragmentHolds = [&] {
        if (fragment == nullptr) return true;
        if (fragment->indirect != gpuIndirect) return false;
        if (!gpuIndirect) return true;
        const auto& indirect = *record.indirect;
        if (!RuntimeSwitches::Set(RuntimeSwitch::ExpressGpuIndirect) || !passIndirect || args->vertexRule != IndirectRule::InPlace || args->instanceRule != IndirectRule::InPlace || indirect.argumentImport == nullptr) return false;
        if (fragment->arguments != indirect.argumentImport->buffer || fragment->argumentOffset != HostImportOffset(*indirect.argumentImport, args->arguments)) return false;
        return !args->countIndirect || (indirect.countImport != nullptr && fragment->count == indirect.countImport->buffer && fragment->countOffset == HostImportOffset(*indirect.countImport, args->countAddress));
    };
    if (fragment != nullptr && (!indirectFragmentHolds() || meshIndirect || captured || meshArguments != nullptr || !record.stencilReplay.empty())) dropFragment(DrawFragmentDrop::Modes);
    auto* batchStream = fragment != nullptr ? CommandStream::FromHandle(commands) : nullptr;
    const bool verifyFragment = fragment != nullptr && record.fragmentVerify && batchStream != nullptr;
    const bool spliceFragment = fragment != nullptr && !record.fragmentVerify && fragmentBindings != nullptr;
    if (fragment != nullptr && record.fragmentVerify && batchStream == nullptr) dropFragment(DrawFragmentDrop::Verify);
    std::uint64_t fragmentMark = 0;
    if (continued) {
        if (verifyFragment) {
            fragmentMark = batchStream->MarkOps();
            recording.Invalidate();
        }
        if (!spliceFragment) record.pipeline->Continue(recording, state, record.masked);
        if (gpuIndirect) {
            noteIndirectArguments(recorder, true, *record.indirect, argumentBuffer, argumentOffset);
            countIndirectInPass();
        }
    } else {
        for (const auto& proxy : record.proxies) {
            proxy->RecordAttachmentProxyLoad(commands, VK_IMAGE_LAYOUT_GENERAL);
            countBarrier(2);
        }
        if (start.barrier) {
            const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | (passIndirect ? static_cast<VkAccessFlags>(VK_ACCESS_INDIRECT_COMMAND_READ_BIT) : VkAccessFlags{0})};
            context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
            countBarrier(1);
            if (passIndirect) recorder->NotePassIndirectReads();
        } else if (movedBehind) {
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT);
            countBarrier(1);
        }
        APS5_LOG_CHARS_OUT_DEBUG("Upload barrier recorded");
        if (inputs.vertexInputBehindGpuWrite) {
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT);
            countBarrier(1);
        }
        const auto pixelLdsSubgroup = PixelFunctionLdsSubgroup(shaders);
        if (pixelLdsSubgroup != 0u) PreparePixelFunctionLds(context, commands, pixelLdsSubgroup);
        if (captured) {
            recordCaptureBarrier(context, commands);
            countBarrier(1);
            meshArguments = state.tessellationCapture->header;
            argumentBuffer = meshArguments->Handle();
        } else if (meshIndirect) {
            meshArguments = recordMeshArguments(context, commands, recorder, true, state, draw, *record.indirect, countBarrier, inputs.restartTable);
            argumentBuffer = meshArguments->Handle();
        } else if (gpuIndirect) {
            rewritten = recordIndirectArguments(context, commands, recorder, true, *record.indirect, scratch, argumentSnapshot, argumentBuffer, argumentOffset, countBarrier, fragment == nullptr);
            if (passIndirect) recorder->NotePassIndirectReads();
        }
        // Earlier recorded work (dispatches, the previous draw) wrote the images in the general
        // layout, which a lean draw renders in: no transitions.
        if (!record.reordered.empty()) {
            std::vector<std::shared_ptr<void>> kept;
            RecordSampleOrder(context, recording, record.reorderPass, record.reordered, true, kept);
            for (auto& object : kept) recorder->Keep(std::move(object));
        }
        for (const auto& write : record.staged) RecordStagedClear(context, commands, *write.temporary);
        APS5_LOG_OUT_DEBUG("Beginning pipeline renderExtent=%ux%u", state.renderExtent.width, state.renderExtent.height);
        const bool optimalDepth = record.proxies.empty() && record.reordered.empty() && !record.depthLocations && beginDepthLayout(context, *recorder, commands, *record.pipeline, *record.framebuffer, state, resources);
        if (spliceFragment) {
            record.pipeline->BeginPass(commands, *record.framebuffer, state.renderExtent, state, false, optimalDepth);
        } else if (verifyFragment) {
            record.pipeline->BeginPass(commands, *record.framebuffer, state.renderExtent, state, false, optimalDepth);
            fragmentMark = batchStream->MarkOps();
            recording.Invalidate();
            record.pipeline->Continue(recording, state, record.masked);
        } else if (optimalDepth) {
            record.pipeline->BeginPass(commands, *record.framebuffer, state.renderExtent, state, false, true);
            record.pipeline->Continue(recording, state, record.masked);
        } else {
            record.pipeline->Begin(recording, *record.framebuffer, state.renderExtent, state, record.masked);
        }
    }
    APS5_LOG_CHARS_OUT_DEBUG("Pipeline Begin OK");
    const bool heapPipeline = record.pipeline->DescriptorHeap();
    NoteHeapRecord(false, heapPipeline);
    if (spliceFragment) {
        if (context.recorder != nullptr) context.recorder->NoteSampledDraw(commands);
        if (batchStream != nullptr) {
            batchStream->Splice(std::move(fragment->commands));
            if (record.fragmentNote != nullptr) record.fragmentNote->spliced = true;
        } else {
            CommandStream::ReplayFragment(fragment->commands, commands);
            if (record.fragmentNote != nullptr) record.fragmentNote->replayed = true;
        }
        recording.Invalidate();
        NoteGuestWorkIn(context, commands);
    } else {
        if (heapPipeline) {
            resources.BindDescriptorHeaps(commands);
            NoteEmptyResourceSet();
        } else if (pushing) {
            resources.PushDrawBindings(commands, record.pipeline->Layout(), *drawBindings, *record.push);
        } else if (drawBindings != nullptr) {
            Require(!resources.HeapStage(), "a GPU-mode draw needs a VK_EXT_descriptor_heap pipeline");
            const auto set = drawBindings->allocation.set;
            const auto fragmentSet = drawBindings->fragmentAllocation.set;
            if (set != VK_NULL_HANDLE) CmdBindDescriptorSets(context, commands, VK_PIPELINE_BIND_POINT_GRAPHICS, record.pipeline->Layout(), 0, 1, &set, SetBindSite::Draw, drawBindings->DynamicCount(0), drawBindings->DynamicOffsets(0));
            if (fragmentSet != VK_NULL_HANDLE) CmdBindDescriptorSets(context, commands, VK_PIPELINE_BIND_POINT_GRAPHICS, record.pipeline->Layout(), ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet, 1, &fragmentSet, SetBindSite::Draw, drawBindings->DynamicCount(1), drawBindings->DynamicOffsets(1));
        } else {
            Require(!resources.HeapStage(), "a GPU-mode draw needs a VK_EXT_descriptor_heap pipeline");
            resources.BindDrawUse(commands, record.pipeline->Layout(), drawUse.get());
        }
        if (!heapPipeline && std::ranges::any_of(shaders, [](const CompiledShader& shader) { return shader.program != nullptr && shader.program->functionLdsBuffer; })) BindPixelFunctionLds(context, commands, record.pipeline->Layout());
        if (!heapPipeline && UsesMeshWindow(shaders)) BindMeshWindow(context, commands, record.pipeline->Layout());
        APS5_LOG_CHARS_OUT_DEBUG("Resources bound");
        std::array<std::byte, PipelinePushConstantBytes> movedPush;
        const std::array<std::byte, PipelinePushConstantBytes>* pushBytes = record.pushBytes;
        VkShaderStageFlags pushStages = record.pushStages;
        if (std::ranges::any_of(record.moved, adjustedInPlace)) {
            std::vector<ShaderResources::MovedBuffer> adjusted;
            for (const auto& entry : record.moved) {
                if (adjustedInPlace(entry)) adjusted.push_back(entry);
            }
            movedPush = pushBytes != nullptr ? *pushBytes : AssemblePushConstants(shaders, RuntimeSwitches::Set(RuntimeSwitch::PushMemcpy));
            if (pushBytes == nullptr) resources.PatchPushConstants(movedPush);
            if (resources.PatchMovedPushConstants(movedPush, adjusted)) {
                if (pushBytes == nullptr) pushStages = PushConstantStages(shaders);
                pushBytes = &movedPush;
            }
        }
        pushDrawConstants(*record.pipeline, commands, state, draw, shaders, resources, pushBytes, pushStages, meshArguments != nullptr ? meshArguments->DeviceAddress() : 0, !inputs.restartTable.empty(), inputs.meshLayerStride, drawBindings.get());
        APS5_LOG_CHARS_OUT_DEBUG("Push constants recorded");
        static const bool timeEachDraw = std::getenv("APS5_PROFILE_GPU_DRAWS") != nullptr;
        auto eachDrawTiming = Recorder::NoTiming;
        if (timeEachDraw) {
            std::uint64_t key = 0;
            for (const auto& shader : shaders) {
                if (shader.program != nullptr && (key == 0 || shader.stage == ShaderRecompiler::ShaderStage::Fragment)) key = shader.program->variantId;
            }
            eachDrawTiming = recorder->BeginInPassGpuTiming(Recorder::DrawTimingKey(key));
        }
        noteDrawFaultToken(context, *recorder, commands, shaders, draw, static_cast<std::uint32_t>(state.stages.mesh.has_value() ? AgcDriver::FaultStage::mesh : AgcDriver::FaultStage::graphics), record.indirect != nullptr ? record.indirect->args : nullptr);
        if (record.stencilReplay.empty()) recordDrawCommands(context, commands, state, draw, inputs, record.indirect, argumentBuffer, argumentOffset, trace, subdraw);
        else {
            recordStencilReplay(context, recording, state, draw, inputs, record.stencilReplay, shaders, resources, pushBytes, pushStages, drawBindings.get());
            for (const auto& [reference, pipeline] : record.stencilReplay) recorder->Keep(pipeline);
        }
        if (record.pipeline->SplitsFaces()) {
            record.pipeline->ContinueBackFaces(recording, state, record.masked);
            recordDrawCommands(context, commands, state, draw, inputs, record.indirect, argumentBuffer, argumentOffset, trace, subdraw);
        }
        if (record.pipeline->SplitsSecondChannel()) {
            record.pipeline->ContinueSecondChannel(recording, state, record.masked);
            recordDrawCommands(context, commands, state, draw, inputs, record.indirect, argumentBuffer, argumentOffset, trace, subdraw);
        }
        recorder->EndGpuTiming(eachDrawTiming);
        if (verifyFragment) {
            auto& note = *record.fragmentNote;
            note.verified = true;
            const char* what = nullptr;
            bool same = CommandStream::SameRecords(fragment->commands, *batchStream, fragmentMark, true, &what);
            if (same) {
                std::vector<std::size_t> chosen;
                chosen.reserve(record.prepared->selected.size());
                for (const auto& item : record.prepared->selected) chosen.push_back(item.allocation);
                std::vector<ShaderResources::BoundBuffer> mine;
                std::vector<ShaderResources::BoundBuffer> today;
                resources.DescribeDrawBindings(fragment->bindings.get(), chosen, record.push, mine, true);
                resources.DescribeDrawBindings(drawBindings.get(), chosen, record.push, today, true);
                same = std::ranges::equal(mine, today, sameFragmentBound);
                if (!same) what = "the bindings";
            }
            note.differs = !same;
            note.what = what;
            note.drop = DrawFragmentDrop::Verify;
        }
    }
    if (meshArguments != nullptr) recorder->Keep(meshArguments);
    if (argumentSnapshot != nullptr) recorder->Keep(argumentSnapshot);
    if (state.tessellationCapture != nullptr) {
        recorder->Keep(state.tessellationCapture->header);
        recorder->Keep(state.tessellationCapture->items);
    }
    if (args != nullptr) CountIndirectDraw(record.indirect->path, record.indirect->readMs, rewritten);
    auto checkRecords = indirectRecordCheck(record.indirect);
    APS5_LOG_CHARS_OUT_DEBUG("Draw recorded");
    std::function<void(CommandState&)> storeProxies;
    if (!continued && (!record.proxies.empty() || !record.reordered.empty() || record.depthLocations)) {
        storeProxies = [proxies = record.proxies, reordered = record.reordered, reorderPass = record.reorderPass, depthLocations = record.depthLocations, context, recorder](CommandState& passRecording) {
            const auto passCommands = passRecording.Commands();
            for (const auto& proxy : proxies) {
                proxy->RecordAttachmentProxyStore(passCommands, VK_IMAGE_LAYOUT_GENERAL);
                Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
            }
            if (!reordered.empty()) {
                std::vector<std::shared_ptr<void>> kept;
                RecordSampleOrder(context, passRecording, reorderPass, reordered, false, kept);
                for (auto& object : kept) recorder->Keep(std::move(object));
            }
            if (depthLocations) {
                RecordDepthSampleLocations(context, passCommands, *depthLocations);
                Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
                recorder->Keep(depthLocations->keep);
            }
        };
    }
    recorder->LeaveRenderPassOpen(passKey, drawTiming, fenced, passAccess, std::move(storeProxies));
    timer.phase(PhaseRecord);
    timer.lap(FullDrawStep::Pass);
    const auto keepBegan = bindingsTimed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    keepRecordedDraw(*recorder, record.resources, std::move(record.pipeline), std::move(record.framebuffer), inputs, std::move(record.targets), std::move(record.targetSlices), state.scissor, std::move(scratch), std::move(checkRecords), record.listed, record.completion, outcome, drawUse, record.spaceKeep != nullptr && *record.spaceKeep != nullptr ? record.spaceKeep->get() : nullptr, record.keeping);
    if (bindingsTimed) record.bindings->keepNanoseconds = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - keepBegan).count());
    timer.phase(PhaseKeep);
    timer.lap(FullDrawStep::Keep);
    if (!record.staged.empty()) {
        const auto merged = recorder->Commands();
        for (auto& write : record.staged) {
            RecordStagedMerge(context, merged, *write.temporary, write.target, write.mask, write.samples, write.area, write.targetOwner);
            recorder->Keep(write.temporary);
        }
    }
    if (record.waited) {
        // The wait the dispatch path makes for such work (source 3, "address-based"): the
        // write-back runs before the next packet, so the CPU never reads copied results or
        // touches a leased allocation before they landed.
        Recorder::CountSync(3);
        const auto ownBefore = timer.profile ? Recorder::ThreadWaitedMs() : 0.0;
        recorder->Sync();
        if (timer.profile) ownWaitedMs += Recorder::ThreadWaitedMs() - ownBefore;
        timer.phase(PhaseSync);
    }
}

// The state with the outputs the pixel shader lacks masked: Vulkan leaves attachments a pixel
// shader has no output for undefined, so their writes are masked (shaders that only store to
// images or buffers keep their targets as they were). Empty when no mask must change.
std::optional<State> maskedState(const State& state, std::uint32_t fragmentOutputs) {
    std::optional<State> masked;
    for (std::size_t index = 0; index < state.blends.size(); ++index) {
        if ((index < 32 && ((fragmentOutputs >> index) & 1u) != 0) || state.blends[index].colorWriteMask == 0) continue;
        if (!masked.has_value()) masked = state;
        masked->blends[index].colorWriteMask = 0;
    }
    return masked;
}

std::optional<DepthSampleLocations> passDepthLocations(const Context& context, const State& state, const Pipeline& pipeline) {
    if (!state.depth || state.depth->samples <= 1u || !state.customSampleLocations || !pipeline.SampleLocationsEnabled()) return std::nullopt;
    auto locations = DepthSurfaceSampleLocations(context, *state.depth);
    locations.grid = pipeline.HostSampleLocationGrid();
    locations.locations = SampleLocations(state);
    return locations;
}

// Debug aid: APS5_DUMP_TARGETS=<n> saves the first n renders of every color target as a raw file
// (u32 width, u32 height, u32 VkFormat, then tightly packed rows).
int DumpTargetLimit() {
    static const int dumpLimit = [] { const char* text = std::getenv("APS5_DUMP_TARGETS"); return text ? std::atoi(text) : 0; }();
    return dumpLimit;
}

// A draw whose targets are all resident is recorded into the device's open batch like a dispatch:
// the CPU does not wait for it, and the drain before it goes away (queue order and the barriers
// keep it behind earlier recorded work); see `recorded` in Draw for the buffers that still need a
// synchronous draw. Other draws keep their own synchronous batch. Debug aid: APS5_SYNC_DRAWS=1
// makes every draw synchronous.
bool RecordDraws() {
    static const bool recordDraws = std::getenv("APS5_SYNC_DRAWS") == nullptr;
    return recordDraws;
}

}

std::optional<std::string> KnownValidationFailure(const Context& context, std::span<const CompiledShader> shaders, const State& state, std::optional<std::uint32_t>* fragmentOutputs) {
    auto& key = validationKeyScratch();
    if (!ValidationKey(context, shaders, state, key)) return std::nullopt;
    std::lock_guard lock(validationMutex());
    const auto& failures = validationFailures();
    if (const auto found = failures.find(key); found != failures.end()) return found->second;
    if (fragmentOutputs != nullptr) {
        const auto& outputs = validationOutputs();
        if (const auto found = outputs.find(key); found != outputs.end()) *fragmentOutputs = found->second;
    }
    return std::nullopt;
}

namespace {

struct TargetBinding {
    ColorTarget color;
    bool gpuTiling = false;
    std::shared_ptr<StorageTexture> resident;
    bool proxied = false;
    std::unique_ptr<Buffer> transfer;
    bool banded = false;
    VkRect2D band{};
    std::size_t bandOffset = 0;
    std::uint32_t bandPitch = 0;
    std::unique_ptr<Buffer> tiled;
    std::unique_ptr<DeviceBuffer> tiledDevice;
    std::unique_ptr<DeviceBuffer> linearDevice;
    std::vector<std::byte> original;
    TileMipLayout mip{};
    std::unique_ptr<RenderTarget> target;
    bool multisampled = false;
    VkImage multisampleImage = VK_NULL_HANDLE;
    std::shared_ptr<void> multisampleOwner;
    std::shared_ptr<StagedTemporary> stagedTemporary;
    VkImageView stagedTarget = VK_NULL_HANDLE;
    const HeapSlotOwner* stagedTargetOwner = nullptr;
    std::shared_ptr<void> stagedTargetKeep;
    std::unique_ptr<Buffer> dump;
};

struct PreparedDraw {
    explicit PreparedDraw(bool profile) : timer(profile) {}
    PreparedDraw(const PreparedDraw&) = delete;
    PreparedDraw& operator=(const PreparedDraw&) = delete;
    std::shared_ptr<const DrawRecipe>* recipeOut = nullptr;
    const DrawTraceContext* trace = nullptr;
    std::uint64_t shape = 0;
    const std::vector<std::uint32_t>* contentKey = nullptr;
    const std::uint32_t* fragmentOutputs = nullptr;
    bool passAttachments = false;
    bool recordRecipe = false;
    bool attachmentsFromMemo = false;
    std::vector<std::uint32_t> stencilReferences;
    std::optional<PerformanceTimer> timing;
    DrawTimer timer;
    DrawOutcome outcome;
    const ShaderResources::BuildTiming* built = nullptr;
    double waitedBefore = 0;
    double ownWaitedMs = 0;
    DrawInputs inputs;
    const Pm4::DrawParameters::IndirectDraw* args = nullptr;
    int dumpLimit = 0;
    std::vector<TargetBinding> targets;
    std::vector<VkImageView> targetViews;
    Recorder* recorder = nullptr;
    ResolvedResources resolved;
    IndirectRecord indirect;
    std::vector<Pm4::DrawArguments> records;
    bool completion = false;
    bool listed = false;
    bool recorded = false;
    bool lean = false;
    bool multisampled = false;
    std::optional<State> masked;
    std::vector<std::pair<std::uint32_t, std::shared_ptr<Pipeline>>> stencilReplay;
    std::shared_ptr<Pipeline> pipeline;
    std::vector<std::shared_ptr<StorageTexture>> owners;
    std::vector<std::shared_ptr<void>> framebufferOwners;
    std::shared_ptr<Framebuffer> framebuffer;
    std::vector<SampleOrderTarget> reordered;
    SampleOrderPass reorderPass;
    std::optional<DepthSampleLocations> depthLocations;
    std::shared_ptr<std::shared_ptr<const void>> spaceKeep;
    DrawCommitView* view = nullptr;
    FullDrawSteps* steps = nullptr;
};

void keepHeldSpace(PreparedDraw& prepared) {
    const auto& resources = prepared.resolved.resources;
    if (prepared.spaceKeep != nullptr || prepared.recorder == nullptr || resources == nullptr || !resources->SpaceReuse() || !resources->SpaceHeld()) return;
    prepared.spaceKeep = reserveSpaceKeep(*prepared.recorder, *resources);
}

bool ShapeMemoEnabled() {
    static const bool enabled = std::getenv("APS5_NO_SHAPE_MEMO") == nullptr;
    return enabled;
}

bool VerifyShapeMemo() {
    static const bool verify = std::getenv("APS5_VERIFY_SHAPE_MEMO") != nullptr;
    return verify;
}

void noteShapeMemoVerified(bool pipeline) {
    static std::atomic<std::uint64_t> targets{0};
    static std::atomic<std::uint64_t> pipelines{0};
    static std::atomic<std::int64_t> last{0};
    (pipeline ? pipelines : targets).fetch_add(1, std::memory_order_relaxed);
    const auto now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto previous = last.load(std::memory_order_relaxed);
    if (now - previous < 10 || !last.compare_exchange_strong(previous, now)) return;
    std::fprintf(stderr, "[shape-memo] verified %llu targets and %llu pipelines\n", static_cast<unsigned long long>(targets.load()), static_cast<unsigned long long>(pipelines.load()));
}

struct ShapeChoice {
    VkImageLayout attachmentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::uint32_t fragmentOutputs = 0;
    std::vector<std::uint64_t> stages;
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;
    std::vector<std::uint32_t> mainKey;
    std::vector<std::uint32_t> fragmentKey;
    std::uint64_t hash = 0;
    std::weak_ptr<Pipeline> pipeline;
};

struct ShapeEntry {
    VkDevice device = VK_NULL_HANDLE;
    std::vector<std::weak_ptr<StorageTexture>> targets;
    std::vector<ShapeChoice> choices;
};

struct ShapeMemoStorage {};
constexpr std::size_t ShapeMemoEntries = 4096;
constexpr std::size_t ShapeMemoChoices = 4;

std::unordered_map<std::uint64_t, ShapeEntry>& shapeMemo() {
    return HostThreadLocal<std::unordered_map<std::uint64_t, ShapeEntry>, ShapeMemoStorage>();
}

const ShapeEntry* findShape(const Context& context, std::uint64_t shape) {
    if (shape == 0 || !ShapeMemoEnabled()) return nullptr;
    const auto& entries = shapeMemo();
    const auto found = entries.find(shape);
    return found == entries.end() || found->second.device != context.device ? nullptr : &found->second;
}

std::shared_ptr<StorageTexture> shapeTarget(const Context& context, std::uint64_t shape, std::size_t index) {
    const auto* entry = findShape(context, shape);
    return entry == nullptr || index >= entry->targets.size() ? nullptr : entry->targets[index].lock();
}

bool sameStages(std::span<const std::uint64_t> kept, std::span<const CompiledShader> shaders) {
    if (kept.size() != shaders.size() * 4u + 1u) return false;
    std::size_t at = 0;
    for (const auto& shader : shaders) {
        const auto identity = shader.program->Identity();
        if (identity.Empty() || kept[at] != static_cast<std::uint64_t>(shader.stage) || kept[at + 1] != identity.low || kept[at + 2] != identity.high || kept[at + 3] != KeyedPushOffset(shader)) return false;
        at += 4;
    }
    return kept[at] == PushConstantStages(shaders);
}

bool sameChoice(const ShapeChoice& choice, const VertexInputLayout& input, const PipelineSetLayouts& layouts, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout, std::uint32_t fragmentOutputs) {
    const auto sameBinding = [](const VkVertexInputBindingDescription& a, const VkVertexInputBindingDescription& b) { return a.binding == b.binding && a.stride == b.stride && a.inputRate == b.inputRate; };
    const auto sameAttribute = [](const VkVertexInputAttributeDescription& a, const VkVertexInputAttributeDescription& b) { return a.location == b.location && a.binding == b.binding && a.format == b.format && a.offset == b.offset; };
    return choice.attachmentLayout == attachmentLayout && choice.fragmentOutputs == fragmentOutputs && std::ranges::equal(choice.bindings, input.bindings, sameBinding) && std::ranges::equal(choice.attributes, input.attributes, sameAttribute) && std::ranges::equal(choice.mainKey, layouts.mainKey) && std::ranges::equal(choice.fragmentKey, layouts.fragmentKey) && sameStages(choice.stages, shaders);
}

std::shared_ptr<Pipeline> shapePipeline(const Context& context, std::uint64_t shape, const VertexInputLayout& input, const PipelineSetLayouts& layouts, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout, std::uint32_t fragmentOutputs) {
    const auto* entry = findShape(context, shape);
    if (entry == nullptr) return nullptr;
    for (const auto& choice : entry->choices) {
        if (!sameChoice(choice, input, layouts, shaders, attachmentLayout, fragmentOutputs)) continue;
        const auto hash = choice.hash;
        const auto expected = choice.pipeline;
        return TouchCachedPipeline(context, hash, expected);
    }
    return nullptr;
}

void rememberShape(const Context& context, std::uint64_t shape, std::span<const TargetBinding> targets, const std::shared_ptr<Pipeline>* pipeline, std::uint64_t hash, const VertexInputLayout& input, const PipelineSetLayouts& layouts, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout, std::uint32_t fragmentOutputs) {
    if (shape == 0 || !ShapeMemoEnabled()) return;
    auto& entries = shapeMemo();
    if (entries.size() >= ShapeMemoEntries && !entries.contains(shape)) entries.clear();
    auto& entry = entries[shape];
    if (entry.device != context.device) {
        entry = {};
        entry.device = context.device;
    }
    entry.targets.resize(targets.size());
    for (std::size_t index = 0; index < targets.size(); ++index) {
        const auto& resident = targets[index].resident;
        if (resident == nullptr || targets[index].color.arrayLastSlice != 0) entry.targets[index].reset();
        else if (entry.targets[index].owner_before(resident) || resident.owner_before(entry.targets[index])) entry.targets[index] = resident;
    }
    if (pipeline == nullptr || hash == 0) return;
    auto& choices = entry.choices;
    const auto kept = std::ranges::find_if(choices, [&](const ShapeChoice& choice) { return sameChoice(choice, input, layouts, shaders, attachmentLayout, fragmentOutputs); });
    if (kept != choices.end()) {
        kept->hash = hash;
        kept->pipeline = *pipeline;
        std::rotate(choices.begin(), kept, std::next(kept));
        return;
    }
    ShapeChoice choice;
    choice.attachmentLayout = attachmentLayout;
    choice.fragmentOutputs = fragmentOutputs;
    for (const auto& shader : shaders) {
        const auto identity = shader.program->Identity();
        if (identity.Empty()) return;
        choice.stages.insert(choice.stages.end(), {static_cast<std::uint64_t>(shader.stage), identity.low, identity.high, KeyedPushOffset(shader)});
    }
    choice.stages.push_back(PushConstantStages(shaders));
    choice.bindings = input.bindings;
    choice.attributes = input.attributes;
    choice.mainKey.assign(layouts.mainKey.begin(), layouts.mainKey.end());
    choice.fragmentKey.assign(layouts.fragmentKey.begin(), layouts.fragmentKey.end());
    choice.hash = hash;
    choice.pipeline = *pipeline;
    if (choices.size() >= ShapeMemoChoices) choices.pop_back();
    choices.insert(choices.begin(), std::move(choice));
}

bool sameColorTarget(const ColorTarget& a, const ColorTarget& b) {
    return a.address == b.address && a.extent.width == b.extent.width && a.extent.height == b.extent.height && a.format == b.format && a.bytes == b.bytes && a.componentMapping == b.componentMapping && a.tileMode == b.tileMode && a.elementBytes == b.elementBytes && a.packing == b.packing && a.dccAddress == b.dccAddress && a.chainDccAddress == b.chainDccAddress && a.dccAlphaOnMsb == b.dccAlphaOnMsb && a.dccPipeAligned == b.dccPipeAligned && a.cmaskAddress == b.cmaskAddress && a.cmaskBytes == b.cmaskBytes && a.cmaskFastClear == b.cmaskFastClear && a.eqaaFragments == b.eqaaFragments && a.surfaceAddress == b.surfaceAddress && a.surfaceExtent.width == b.surfaceExtent.width && a.surfaceExtent.height == b.surfaceExtent.height && a.mipCount == b.mipCount && a.mip == b.mip && a.mipTail == b.mipTail && a.clearWords == b.clearWords && a.slot == b.slot && a.depth == b.depth && a.depthSlice == b.depthSlice && a.arrayAddress == b.arrayAddress && a.arrayLastSlice == b.arrayLastSlice && a.arraySlice == b.arraySlice && a.layers == b.layers && a.exportIndex == b.exportIndex && a.uintExport == b.uintExport && a.volume == b.volume && a.pipeBankXor == b.pipeBankXor && a.samples == b.samples && a.coverageSamples == b.coverageSamples && a.dccControl == b.dccControl && a.fmaskAddress == b.fmaskAddress && a.fmaskPipeBankXor == b.fmaskPipeBankXor && a.partialWriteMask == b.partialWriteMask && a.staged == b.staged && a.stagedMask == b.stagedMask && a.stagedFormat == b.stagedFormat;
}

bool sameDepthTarget(const DepthTarget& a, const DepthTarget& b) {
    return a.address == b.address && a.stencilAddress == b.stencilAddress && a.extent.width == b.extent.width && a.extent.height == b.extent.height && a.format == b.format && std::bit_cast<std::uint32_t>(a.clearDepth) == std::bit_cast<std::uint32_t>(b.clearDepth) && a.clearStencil == b.clearStencil && a.htileAddress == b.htileAddress && a.htileStencil == b.htileStencil && a.samples == b.samples && a.depthSwizzle == b.depthSwizzle && a.pipeBankXor == b.pipeBankXor && a.firstSlice == b.firstSlice && a.lastSlice == b.lastSlice && a.iterate256 == b.iterate256 && a.htilePipeAligned == b.htilePipeAligned && a.htileVrsEncoding == b.htileVrsEncoding && a.htileDepthOnly == b.htileDepthOnly && a.gcn == b.gcn;
}

struct AttachmentMemo {
    bool valid = false;
    VkDevice device = VK_NULL_HANDLE;
    std::uint64_t recorder = 0;
    std::uint64_t generation = 0;
    std::uint64_t epoch = 0;
    std::uint64_t proof = 0;
    std::uint64_t pending = 0;
    std::uint64_t depthSerial = 0;
    std::uint64_t depthReleased = 0;
    std::vector<ColorTarget> colors;
    std::vector<std::weak_ptr<StorageTexture>> residents;
    std::vector<std::uint64_t> versions;
    std::vector<bool> gpuTiling;
    std::vector<bool> proxied;
    std::vector<bool> multisampled;
    std::vector<VkImageView> views;
    std::optional<DepthTarget> depth;
    bool depthWrite = false;
    bool stencilTest = false;
    bool stencilWritten = false;
    VkImageView depthView = VK_NULL_HANDLE;
    bool multisample = false;
    std::shared_ptr<const std::atomic<bool>> depthActive;
};

struct AttachmentMemoStorage {};

AttachmentMemo& attachmentMemo() {
    return HostThreadLocal<AttachmentMemo, AttachmentMemoStorage>();
}

bool PassAttachmentMemoEnabled() {
    return !RuntimeSwitches::Set(RuntimeSwitch::NoPassAttachmentMemo) && ShaderResources::EpochProofs();
}

bool VerifyPassAttachments() {
    static const bool verify = std::getenv("APS5_VERIFY_PASS_ATTACHMENTS") != nullptr;
    return verify;
}

bool CheckedPassAttachments() {
    if (!VerifyPassAttachments()) return false;
    const auto& memo = attachmentMemo();
    return memo.valid && memo.depthActive != nullptr && RuntimeSwitches::Set(RuntimeSwitch::MsaaPassMemo);
}

enum class AttachmentEvent : std::size_t { Hit, NoMemo, Pass, Serials, Targets, Images, Depth, Stored, Ineligible, Verified, Mismatch, MultisampleDepth, MultisampleColor, Count };

void noteAttachments(AttachmentEvent event) {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr || std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!stats) return;
    static std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(AttachmentEvent::Count)> counts{};
    static std::atomic<std::int64_t> reportedAt{0};
    counts[static_cast<std::size_t>(event)].fetch_add(1, std::memory_order_relaxed);
    const auto tenSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (reportedAt.exchange(tenSeconds, std::memory_order_relaxed) == tenSeconds) return;
    std::array<unsigned long long, static_cast<std::size_t>(AttachmentEvent::Count)> taken{};
    for (std::size_t index = 0; index < taken.size(); ++index) taken[index] = counts[index].exchange(0, std::memory_order_relaxed);
    AgcDriver::ProfilePrint_nid_no_patch("[pass-attachments] 10 s: %llu hits; misses: %llu no memo, %llu pass, %llu serials, %llu targets, %llu images, %llu depth; %llu stored, %llu ineligible (%llu multisample depth, %llu multisampled colour); verify %llu checked, %llu differed\n", taken[0], taken[1], taken[2], taken[3], taken[4], taken[5], taken[6], taken[7], taken[8], taken[11], taken[12], taken[9], taken[10]);
}

void checkMultisampleDepth(const Context& context, const State& state) {
    const auto& memo = attachmentMemo();
    const bool same = RecheckDepthSurfaceView(context, *state.depth, state.stencilTest) == memo.depthView;
    noteAttachments(same ? AttachmentEvent::Verified : AttachmentEvent::Mismatch);
    if (!same) {
        static std::atomic<int> reports{0};
        if (reports.fetch_add(1, std::memory_order_relaxed) < 20) std::fprintf(stderr, "[pass-attachments] verify: the memo's multisample depth view differs from its surface's, or the surface had a clear to apply (target 0x%llx, depth 0x%llx)\n", static_cast<unsigned long long>(state.hasColorTarget ? state.color.address : 0), static_cast<unsigned long long>(state.depth->address));
    }
}

bool fillMultisampleAttachments(const Context& context, const State& state, PreparedDraw& prepared) {
    const auto& memo = attachmentMemo();
    auto& targets = prepared.targets;
    auto& targetViews = prepared.targetViews;
    for (std::size_t index = 0; index < state.colors.size(); ++index) {
        auto& binding = targets[index];
        binding.color = state.colors[index];
        binding.gpuTiling = memo.gpuTiling[index];
        if (memo.multisampled[index]) {
            binding.multisampled = true;
            targetViews.push_back(MultisampleTargetView(context, binding.color, binding.multisampleOwner, &binding.multisampleImage));
            continue;
        }
        binding.proxied = memo.proxied[index];
        targetViews.push_back(memo.views[index]);
    }
    if (state.depth && ClaimDepthTargetMemory(*state.depth)) {
        noteAttachments(AttachmentEvent::Depth);
        targetViews.push_back(DepthSurfaceViewWithStencilWriteIntent(context, *state.depth, state.depthWrite, state.stencilTest, writesStencilTarget(state), overwrittenStencil(state)));
        return true;
    }
    noteAttachments(AttachmentEvent::Hit);
    if (!state.depth) return true;
    targetViews.push_back(memo.depthView);
    if (memo.depthActive != nullptr && VerifyPassAttachments()) checkMultisampleDepth(context, state);
    return true;
}

bool memoAttachments(const Context& context, const State& state, PreparedDraw& prepared, bool fill) {
    const auto& memo = attachmentMemo();
    const auto miss = [](AttachmentEvent why) { noteAttachments(why); return false; };
    if (!memo.valid || memo.device != context.device || (memo.multisample && !RuntimeSwitches::Set(RuntimeSwitch::MsaaPassMemo))) return miss(AttachmentEvent::NoMemo);
    const auto* recorder = Recorder::Active();
    if (recorder == nullptr || recorder->Id() != memo.recorder || !recorder->RenderPassOpen() || recorder->RenderPassGeneration() != memo.generation) return miss(AttachmentEvent::Pass);
    if (GuestMemory::CollectEpoch() != memo.epoch || GuestMemory::ProofSerial() != memo.proof || StorageTexture::PendingSerial() != memo.pending) return miss(AttachmentEvent::Serials);
    if (state.colors.size() != memo.colors.size()) return miss(AttachmentEvent::Targets);
    for (std::size_t index = 0; index < state.colors.size(); ++index) {
        if (!sameColorTarget(state.colors[index], memo.colors[index])) return miss(AttachmentEvent::Targets);
    }
    if (state.depth.has_value() != memo.depth.has_value()) return miss(AttachmentEvent::Depth);
    if (state.depth && overwrittenStencil(state)) return miss(AttachmentEvent::Depth);
    if (state.depth) {
        if (!sameDepthTarget(*state.depth, *memo.depth) || state.depthWrite != memo.depthWrite || state.stencilTest != memo.stencilTest || writesStencilTarget(state) != memo.stencilWritten || DepthSurfacesSerial() != memo.depthSerial || DepthSurfacesReleased() != memo.depthReleased || (memo.depthActive != nullptr && !memo.depthActive->load(std::memory_order_acquire))) return miss(AttachmentEvent::Depth);
    }
    auto& targets = prepared.targets;
    for (std::size_t index = 0; index < memo.residents.size(); ++index) {
        if (memo.multisampled[index]) continue;
        auto resident = memo.residents[index].lock();
        if (resident == nullptr || resident->Version() != memo.versions[index] || !StorageImageCached(context, resident.get()) || !StorageImageServesKeys(*resident, SurfaceForTarget(state.colors[index]).dccAddress)) {
            for (std::size_t filled = 0; filled < index; ++filled) targets[filled].resident.reset();
            return miss(AttachmentEvent::Images);
        }
        if (fill) targets[index].resident = std::move(resident);
    }
    if (!fill) return true;
    if (memo.multisample) return fillMultisampleAttachments(context, state, prepared);
    noteAttachments(AttachmentEvent::Hit);
    auto& targetViews = prepared.targetViews;
    for (std::size_t index = 0; index < state.colors.size(); ++index) {
        auto& binding = targets[index];
        binding.color = state.colors[index];
        binding.gpuTiling = memo.gpuTiling[index];
        binding.proxied = memo.proxied[index];
        targetViews.push_back(memo.views[index]);
    }
    if (state.depth) targetViews.push_back(memo.depthView);
    return true;
}

void checkAttachments(const State& state, const PreparedDraw& prepared) {
    const auto& memo = attachmentMemo();
    bool same = prepared.targetViews.size() == memo.views.size() + (memo.depth ? 1u : 0u);
    for (std::size_t index = 0; same && index < memo.views.size(); ++index) same = memo.multisampled[index] || (prepared.targets[index].resident == memo.residents[index].lock() && prepared.targetViews[index] == memo.views[index]);
    if (same && state.depth) same = prepared.targetViews.back() == memo.depthView;
    noteAttachments(same ? AttachmentEvent::Verified : AttachmentEvent::Mismatch);
    if (!same) {
        static std::atomic<int> reports{0};
        if (reports.fetch_add(1, std::memory_order_relaxed) < 20) std::fprintf(stderr, "[pass-attachments] verify: the memo's attachments differ from the full lookup (target 0x%llx, depth 0x%llx)\n", static_cast<unsigned long long>(state.hasColorTarget ? state.color.address : 0), static_cast<unsigned long long>(state.depth ? state.depth->address : 0));
    }
}

void rememberAttachments(const Context& context, const State& state, const PreparedDraw& prepared) {
    auto& memo = attachmentMemo();
    const auto* recorder = prepared.recorder;
    const auto epoch = GuestMemory::CollectEpoch();
    const bool multisamplePasses = RuntimeSwitches::Set(RuntimeSwitch::MsaaPassMemo);
    const bool multisampledDepth = state.depth && state.depth->samples != 1u;
    const bool multisampledColors = multisamplePasses && std::ranges::all_of(prepared.targets, [](const TargetBinding& binding) { return binding.multisampled && binding.multisampleOwner != nullptr && binding.resident == nullptr && binding.target == nullptr && binding.stagedTemporary == nullptr && binding.dump == nullptr && !binding.color.staged && binding.color.partialWriteMask == 0; });
    const bool eligible = epoch != 0 && recorder != nullptr && recorder->RenderPassOpen() && prepared.dumpLimit == 0 && (!multisampledDepth || multisamplePasses) && prepared.targetViews.size() == prepared.targets.size() + (state.depth ? 1u : 0u) && (multisampledColors || std::ranges::all_of(prepared.targets, [](const TargetBinding& binding) { return !binding.multisampled && binding.resident != nullptr && binding.target == nullptr && binding.stagedTemporary == nullptr && binding.dump == nullptr && !binding.color.staged && binding.color.partialWriteMask == 0; }));
    std::shared_ptr<const std::atomic<bool>> depthActive;
    if (eligible && multisampledDepth) depthActive = prepared.attachmentsFromMemo && memo.valid && memo.depthActive != nullptr && memo.depthView == prepared.targetViews.back() ? memo.depthActive : DepthSurfaceActive(context, *state.depth, prepared.targetViews.back());
    if (!eligible || (multisampledDepth && depthActive == nullptr)) {
        memo.valid = false;
        noteAttachments(AttachmentEvent::Ineligible);
        if (state.depth && state.depth->samples != 1u) noteAttachments(AttachmentEvent::MultisampleDepth);
        if (std::ranges::any_of(prepared.targets, [](const TargetBinding& binding) { return binding.multisampled; })) noteAttachments(AttachmentEvent::MultisampleColor);
        return;
    }
    memo.valid = true;
    memo.device = context.device;
    memo.recorder = recorder->Id();
    memo.generation = recorder->RenderPassGeneration();
    memo.epoch = epoch;
    memo.proof = GuestMemory::ProofSerial();
    memo.pending = StorageTexture::PendingSerial();
    memo.depthSerial = DepthSurfacesSerial();
    memo.depthReleased = DepthSurfacesReleased();
    memo.colors.assign(state.colors.begin(), state.colors.end());
    memo.residents.clear();
    memo.versions.clear();
    memo.gpuTiling.clear();
    memo.proxied.clear();
    memo.multisampled.clear();
    memo.views.clear();
    for (std::size_t index = 0; index < prepared.targets.size(); ++index) {
        const auto& binding = prepared.targets[index];
        memo.residents.emplace_back(binding.resident);
        memo.versions.push_back(binding.multisampled ? 0u : binding.resident->Version());
        memo.gpuTiling.push_back(binding.gpuTiling);
        memo.proxied.push_back(binding.proxied);
        memo.multisampled.push_back(binding.multisampled);
        memo.views.push_back(prepared.targetViews[index]);
    }
    memo.multisample = multisampledDepth || std::ranges::any_of(prepared.targets, [](const TargetBinding& binding) { return binding.multisampled; });
    memo.depthActive = depthActive;
    memo.depth = state.depth;
    memo.depthWrite = state.depthWrite;
    memo.stencilTest = state.stencilTest;
    memo.stencilWritten = state.depth ? writesStencilTarget(state) : false;
    memo.depthView = state.depth ? prepared.targetViews.back() : VK_NULL_HANDLE;
    noteAttachments(AttachmentEvent::Stored);
}

bool storesIntoResident(const ShaderResources& resources, const StorageTexture& resident) {
    return std::ranges::any_of(resources.StorageImages(), [&](const std::pair<VkImage, bool>& image) { return image.first == resident.Image() && image.second; });
}

ReplayReason heapReplayDrawReason(const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, const DrawInputs& inputs, std::span<const TargetBinding> targets, bool stencil, bool recordable) {
    static const bool drawTransitions = std::getenv("APS5_DRAW_TRANSITIONS") != nullptr;
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!recordable) return ReplayReason::NotRecordable;
    if (drawTransitions) return ReplayReason::Transitions;
    if (stencil) return ReplayReason::Stencil;
    if (draw.indirect) return ReplayReason::Indirect;
    if (state.stages.mesh) return ReplayReason::Mesh;
    if (state.stages.tessellation || state.tessellationCapture != nullptr) return ReplayReason::Tessellation;
    if (state.quadList) return ReplayReason::QuadList;
    if (StripPatches(state, shaders)) return ReplayReason::StripPatches;
    if (inputs.listRestartPrimitive != 0 || !inputs.restartTable.empty()) return ReplayReason::Restart;
    if (std::ranges::any_of(targets, [](const TargetBinding& binding) { return binding.multisampled || binding.proxied || binding.stagedTemporary != nullptr || binding.color.partialWriteMask != 0; })) return ReplayReason::Targets;
    if (CaptureInputsEnabled()) return ReplayReason::Capture;
    if (profile) return ReplayReason::Profile;
    if (!GuestMemory::OnProofThread()) return ReplayReason::Worker;
    return ReplayReason::Eligible;
}

bool VerifyTargetMemo() {
    static const bool verify = std::getenv("APS5_VERIFY_TARGET_MEMO") != nullptr;
    return verify;
}

bool TargetMemoEnabled() {
    static const bool cached = std::getenv("APS5_NO_TEXTURE_CACHE") == nullptr;
    return cached && !RuntimeSwitches::Set(RuntimeSwitch::NoTargetMemo) && ShaderResources::EpochProofs();
}

struct TargetProof {
    std::uint64_t epoch = 0;
    std::uint64_t proofs = 0;
    std::uint64_t pending = 0;
    std::uint64_t unwatch = 0;
    std::uint64_t allocations = 0;
    std::uint64_t surfaces = 0;
    std::uint64_t released = 0;
    std::uint64_t claims = 0;
    const Recorder* recorder = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    bool operator==(const TargetProof&) const = default;
    const char* MovedBefore(const TargetProof& after) const {
        if (epoch != after.epoch) return "epoch";
        if (surfaces != after.surfaces) return "surfaces";
        return released != after.released ? "released" : nullptr;
    }
};

TargetProof currentTargetProof(const Context& context) {
    return {GuestMemory::CollectEpoch(), GuestMemory::ProofSerial(), StorageTexture::PendingSerial(), GuestMemory::UnwatchSerial(), GuestAllocations::GuestAllocationsGeneration_nid_postfix(), DepthSurfacesSerial(), DepthSurfacesReleased(), MultisampleClaimSerial(), Recorder::Active(), context.device};
}


struct DepthFootprint {
    std::uint64_t address = 0;
    std::uint64_t stencilAddress = 0;
    std::uint64_t htileAddress = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t samples = 0;
    bool d16 = false;
    bool sliced = false;
    bool operator==(const DepthFootprint&) const = default;
};

std::optional<DepthFootprint> depthFootprint(const std::optional<DepthTarget>& depth) {
    if (!depth) return std::nullopt;
    return DepthFootprint{depth->address, depth->stencilAddress, depth->htileAddress, depth->extent.width, depth->extent.height, depth->samples,
        depth->format == VK_FORMAT_D16_UNORM || depth->format == VK_FORMAT_D16_UNORM_S8_UINT, depth->firstSlice != 0u || depth->lastSlice != 0u};
}

struct ColorMemo {
    struct Attachment {
        std::weak_ptr<StorageTexture> resident;
        bool proxied = false;
        VkImageView view = VK_NULL_HANDLE;
    };
    bool valid = false;
    TargetProof proof;
    std::vector<ColorTarget> descriptors;
    std::vector<Attachment> attachments;
    std::optional<DepthFootprint> clearOfDepth;
};

struct DepthMemo {
    bool valid = false;
    TargetProof proof;
    std::optional<DepthTarget> descriptor;
};

struct TargetMemo {
    ColorMemo colors;
    DepthMemo depth;
    void Invalidate() {
        colors.valid = false;
        depth.valid = false;
    }
};

struct TargetMemoStorage {};
TargetMemo& ThreadTargetMemo() {
    return HostThreadLocal<TargetMemo, TargetMemoStorage>();
}
thread_local std::uint64_t targetMemoBuilds = 0;
std::atomic<const TargetMemo*> targetMemoOwner{nullptr};

constexpr std::array<const char*, static_cast<std::size_t>(TargetMemoEvent::Count)> TargetMemoEventNames{
    "lookup", "hit", "overlap cached", "overlap computed", "accept", "accept without depth", "depth hit", "avoided refresh lookups",
    "invalid", "thread changed", "descriptor", "resident gone", "resident evicted", "depth overlap", "metadata",
    "epoch", "proofs", "pending", "unwatch", "allocations", "surfaces", "released", "claims", "recorder", "device",
    "post-depth invalidated", "depth fallback", "admission", "recipe invalidated", "build", "verify checked"
};

std::array<ProfileCounter, static_cast<std::size_t>(TargetMemoEvent::Count)>& targetMemoEvents() {
    static std::array<ProfileCounter, static_cast<std::size_t>(TargetMemoEvent::Count)> events;
    return events;
}

const bool targetMemoProfiled = std::getenv("APS5_PROFILE_TARGET_MEMO") != nullptr || std::getenv("APS5_PROFILE_DRAW") != nullptr;

void noteTargetMemo(TargetMemoEvent event, std::uint64_t count = 1) {
    if (targetMemoProfiled) targetMemoEvents()[static_cast<std::size_t>(event)].Add(count);
#ifdef TRACY_ENABLE
    static std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(TargetMemoEvent::Count)> counts{};
    const auto index = static_cast<std::size_t>(event);
    TracyCPlotI(TargetMemoEventNames[index], static_cast<std::int64_t>(counts[index].fetch_add(count, std::memory_order_relaxed) + count));
#endif
}

void reportTargetMemo() {
    using Clock = std::chrono::steady_clock;
    static std::atomic<std::int64_t> reported{Clock::now().time_since_epoch().count()};
    static std::array<std::uint64_t, static_cast<std::size_t>(TargetMemoEvent::Count)> previous{};
    const auto now = Clock::now().time_since_epoch().count();
    auto last = reported.load(std::memory_order_relaxed);
    if (Clock::duration(now - last) < std::chrono::seconds(10) || !reported.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    std::string line;
    for (std::size_t index = 0; index < previous.size(); ++index) {
        const auto total = targetMemoEvents()[index].Sum();
        const auto count = total - previous[index];
        previous[index] = total;
        if (count == 0) continue;
        line += (line.empty() ? " " : ", ") + std::string(TargetMemoEventNames[index]) + " " + std::to_string(count);
    }
    AgcDriver::ProfilePrint_nid_no_patch("[target-memo] (10 s, all threads):%s\n", line.empty() ? " none" : line.c_str());
}

bool colorProofMatches(const TargetProof& before, const TargetProof& after) {
    if (before == after) return true;
    const auto note = [&](bool changed, TargetMemoEvent event) { if (changed) noteTargetMemo(event); };
    note(before.epoch != after.epoch, TargetMemoEvent::Epoch);
    note(before.proofs != after.proofs, TargetMemoEvent::Proofs);
    note(before.pending != after.pending, TargetMemoEvent::Pending);
    note(before.unwatch != after.unwatch, TargetMemoEvent::Unwatch);
    note(before.allocations != after.allocations, TargetMemoEvent::Allocations);
    note(before.surfaces != after.surfaces, TargetMemoEvent::Surfaces);
    note(before.released != after.released, TargetMemoEvent::Released);
    note(before.claims != after.claims, TargetMemoEvent::Claims);
    note(before.recorder != after.recorder, TargetMemoEvent::Recorder);
    note(before.device != after.device, TargetMemoEvent::Device);
    return false;
}


bool ownsTargetMemo(TargetMemo& memo) {
    if (targetMemoOwner.load(std::memory_order_relaxed) == &memo) return true;
    targetMemoOwner.store(&memo, std::memory_order_relaxed);
    if (memo.colors.valid || memo.depth.valid) noteTargetMemo(TargetMemoEvent::ThreadChanged);
    memo.Invalidate();
    return false;
}

void dropTargetMemo() {
    if (targetMemoOwner.load(std::memory_order_relaxed) == nullptr) return;
    ThreadTargetMemo().Invalidate();
    targetMemoOwner.store(nullptr, std::memory_order_relaxed);
}

bool colorOverlapsDepth(const State& state, const TargetBinding& binding) {
    if (!state.depth) return false;
    const auto& color = binding.color;
    const auto& resident = *binding.resident;
    const auto& descriptor = resident.Descriptor();
    return DepthTargetOverlaps(*state.depth, resident.Descriptor().baseAddress, resident.GuestBytes()) ||
        (descriptor.dccAddress != 0 && DepthTargetOverlaps(*state.depth, descriptor.dccAddress, std::max<std::uint64_t>(resident.GuestBytes() / 256u, 1u))) ||
        (color.dccAddress != 0 && DepthTargetOverlaps(*state.depth, color.dccAddress, std::max<std::size_t>(colorKeyCount(color, color.bytes), 1u))) ||
        (color.chainDccAddress != 0 && DepthTargetOverlaps(*state.depth, color.chainDccAddress, std::max<std::size_t>(colorKeyCount(color, color.bytes), 1u))) ||
        (color.cmaskAddress != 0 && DepthTargetOverlaps(*state.depth, color.cmaskAddress, color.cmaskBytes));
}

bool metadataProved(const ColorTarget& color) {
    const auto keys = std::max<std::size_t>(colorKeyCount(color, color.bytes), 1u);
    const bool proved = (color.dccAddress == 0 || GuestMemory::CollectWrites(color.dccAddress, keys) != 0) &&
        (color.chainDccAddress == 0 || GuestMemory::CollectWrites(color.chainDccAddress, keys) != 0) &&
        (color.cmaskAddress == 0 || cmaskProved(color.cmaskAddress, color.cmaskBytes));
    if (!proved) noteTargetMemo(TargetMemoEvent::Metadata);
    return proved;
}

void rememberTargets(const State& state, std::span<const TargetBinding> targets, std::span<const VkImageView> views, const TargetProof& proof) {
    ++targetMemoBuilds;
    noteTargetMemo(TargetMemoEvent::Build);
    auto& colors = ThreadTargetMemo().colors;
    auto& depth = ThreadTargetMemo().depth;
    const bool enabled = TargetMemoEnabled() && proof.epoch != 0;
    colors.valid = enabled && !targets.empty() && std::ranges::all_of(targets, [&](const TargetBinding& binding) {
        return binding.resident != nullptr && binding.resident->Cached() && !binding.multisampled && binding.color.fmaskAddress == 0 && !colorOverlapsDepth(state, binding) && metadataProved(binding.color);
    });
    if (colors.valid) {
        colors.proof = proof;
        colors.clearOfDepth = depthFootprint(state.depth);
        colors.descriptors.assign(state.colors.begin(), state.colors.end());
        colors.attachments.resize(targets.size());
        for (std::size_t index = 0; index < targets.size(); ++index) colors.attachments[index] = {targets[index].resident, targets[index].proxied, views[index]};
    }
    depth.valid = enabled && (!state.depth || state.depth->samples == 1u);
    if (depth.valid) {
        depth.proof = proof;
        depth.descriptor = state.depth;
    }
}

bool repeatColors(const Context& context, const State& state, const TargetProof& proof, std::vector<TargetBinding>& targets, std::vector<VkImageView>& views) {
    auto& memo = ThreadTargetMemo().colors;
    thread_local std::uint32_t lookups = 0;
    if (targetMemoProfiled && (++lookups & 4095u) == 0u) reportTargetMemo();
    if (state.colors.empty()) return false;
    noteTargetMemo(TargetMemoEvent::Lookup);
    if (!memo.valid || !TargetMemoEnabled()) {
        noteTargetMemo(TargetMemoEvent::Invalid);
        return false;
    }
    if (!colorProofMatches(memo.proof, proof)) return false;
    if (memo.descriptors != state.colors) {
        noteTargetMemo(TargetMemoEvent::Descriptor);
        return false;
    }
    const auto footprint = depthFootprint(state.depth);
    const bool overlapKnown = footprint == memo.clearOfDepth && !RuntimeSwitches::Set(RuntimeSwitch::NoTargetMemoOverlapCache);
    for (std::size_t index = 0; index < targets.size(); ++index) {
        auto& binding = targets[index];
        binding.resident = memo.attachments[index].resident.lock();
        if (binding.resident == nullptr || !binding.resident->Cached()) {
            noteTargetMemo(binding.resident == nullptr ? TargetMemoEvent::ResidentGone : TargetMemoEvent::ResidentEvicted);
            return false;
        }
        binding.color = state.colors[index];
        if (VerifyTargetMemo()) Require(overlapKnown ? !colorOverlapsDepth(state, binding) : true, "APS5_VERIFY_TARGET_MEMO: a cached depth overlap differs from its computation");
        if (!overlapKnown && colorOverlapsDepth(state, binding)) {
            noteTargetMemo(TargetMemoEvent::DepthOverlap);
            return false;
        }
        binding.gpuTiling = binding.color.tileMode == ColorTileMode::RenderTarget && context.detiler != nullptr;
        binding.proxied = memo.attachments[index].proxied;
        views.push_back(memo.attachments[index].view);
    }
    noteTargetMemo(TargetMemoEvent::Hit);
    noteTargetMemo(overlapKnown ? TargetMemoEvent::OverlapCached : TargetMemoEvent::OverlapComputed);
    if (!overlapKnown) memo.clearOfDepth = footprint;
    return true;
}

VkImageView repeatDepth(const Context& context, const State& state, const TargetProof* proof) {
    const auto& memo = ThreadTargetMemo().depth;
    if (!state.depth || !memo.valid || !TargetMemoEnabled() || targetMemoOwner.load(std::memory_order_relaxed) != &ThreadTargetMemo() || memo.proof != (proof != nullptr ? *proof : currentTargetProof(context)) || !memo.descriptor) return VK_NULL_HANDLE;
    if (*memo.descriptor != *state.depth) return VK_NULL_HANDLE;
    return RepeatDepthSurfaceView(context, *state.depth, state.depthWrite, writesStencilTarget(state));
}

const char* targetObligation(const Context& context, const State& state, std::span<const TargetBinding> targets, bool repeatedDepth) {
    for (const auto& binding : targets) {
        const auto& color = binding.color;
        const auto layers = color.arrayLastSlice != 0 ? color.layers : 0u;
        if (const auto* owed = binding.resident->RefreshObligation(layers != 0 ? color.arraySlice : 0u, layers)) return owed;
        if (const auto* owed = DepthPlanesObligation(context.device, color.MemoryAddress(), color.bytes)) return owed;
        if (color.cmaskAddress == 0 || cmaskProved(color.cmaskAddress, color.cmaskBytes)) continue;
        std::vector<std::uint32_t> cleared;
        if (readCmask(color, CmaskLayout(color.extent.width, color.extent.height), cleared) != CmaskState::Expanded) return "a CMASK clear";
    }
    return repeatedDepth ? DepthTargetObligation(context, *state.depth, state.depthWrite) : nullptr;
}

bool prepareDraw(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, PreparedDraw& prepared) {
    auto*& recipeOut = prepared.recipeOut;
    auto& stencilReferences = prepared.stencilReferences;
    for (const auto& shader : shaders) {
        if (shader.program != nullptr && !shader.program->stencilReferences.empty()) {
            Require(shader.stage == ShaderRecompiler::ShaderStage::Fragment && stencilReferences.empty(), "stencil replay requires exactly one fragment stage");
            stencilReferences = shader.program->stencilReferences;
        }
    }
    if (!stencilReferences.empty()) {
        Require(!draw.indirect && !state.primitiveRestart && !state.rectList && !state.quadList && !state.stages.mesh && !state.stages.tessellation && state.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, "stencil replay supports direct triangle lists without restart, mesh or tessellation");
        Require(!state.depthBiasPerFace && !ShadersWriteMemory(shaders), "stencil replay refuses face splits and shader memory writes");
        Require(std::ranges::all_of(shaders, [](const CompiledShader& shader) { return shader.program != nullptr && shader.program->stencilReplaySafe && (shader.stage == ShaderRecompiler::ShaderStage::Vertex || shader.stage == ShaderRecompiler::ShaderStage::Fragment); }), "stencil replay refuses stages with side effects, clocks, subgroup dependencies or primitive ID inputs");
        if (stencilReferences.size() == 256u) {
            static std::once_flag unbounded;
            std::call_once(unbounded, [] { APS5_LOG_CHARS_OUT("Stencil replay: a draw replays all 256 stencil reference values because its pixel shader's export is not bounded, recording 256 passes per triangle and instance"); });
        }
        for (const auto& shader : shaders) {
            if (shader.stage != ShaderRecompiler::ShaderStage::Vertex) continue;
            Require((shader.program->hostSubgroupSize == 1u || shader.program->hostSubgroupSize == context.subgroup.subgroupSize) && shader.program->spirv.size() >= 5u && shader.program->spirv[1] >= 0x00010000u && shader.program->spirv[1] < 0x00010600u, "stencil replay vertex subgroup contract differs from the physical device");
        }
        Require(static_cast<std::uint64_t>(draw.firstInstance) + draw.instanceCount <= 0x100000000ull && (draw.indexed || static_cast<std::uint64_t>(draw.firstVertex) + draw.indexCount <= 0x100000000ull), "stencil replay draw indices overflow");
        Require(state.samples == 1u && (!state.depth || HasStencil(state.depth->format)), "stencil replay requires single-sample rendering and an eight-bit stencil format");
        recipeOut = nullptr;
    }
    if (state.quadList || StripPatches(state, shaders)) recipeOut = nullptr;
    if (context.recorder != nullptr) context.recorder->BoundKeptBytes();
    // A GPU-mode (descriptor heap) draw gets a recipe only as a data recipe, whose hits write the
    // draw's heap stage words as a template hit does (APS5_NO_HEAP_RECIPES turns it off).
    if (AnyHeapShader(shaders) && (RuntimeSwitches::Set(RuntimeSwitch::NoHeapRecipes) || RuntimeSwitches::Set(RuntimeSwitch::NoDataRecipes))) recipeOut = nullptr;
    auto& timing = prepared.timing.emplace("Graphics.Draw");
    // APS5_PROFILE_DRAW prints the time of each phase of the draw (microseconds) and the [draws] totals.
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    auto& timer = prepared.timer;
    timer = DrawTimer(profile);
    timer.steps = prepared.steps;
    // APS5_TRACE_DRAWS (together with APS5_PROFILE_DRAW; alone it only enables Driver.cpp's own
    // [draw] lines) additionally prints one line per draw with its phases and, for synchronous
    // draws, their inputs: per-draw string building at ~1400 draws per 10 s is itself a cost the
    // totals must not carry.
    static const bool traceDraws = std::getenv("APS5_TRACE_DRAWS") != nullptr;
    auto& outcome = prepared.outcome;
    auto& built = prepared.built;
    // The thread's GPU waits so far: the difference at the report, less the draw's own syncs
    // (`ownWaitedMs`, sampled around them), is what this draw waited for through the flush hook
    // (nested in the draw's hold, see DrawOutcome::hookWaitUs).
    const auto waitedBefore = prepared.waitedBefore = profile ? Recorder::ThreadWaitedMs() : 0.0;
    auto& ownWaitedMs = prepared.ownWaitedMs;
    const auto report = [&](const char* suffix) { reportDrawEnd(state, timer, built, outcome, waitedBefore, ownWaitedMs, suffix); };
    auto& inputs = prepared.inputs;
    const auto plan = planDraw(context, state, draw, shaders, nullptr, outcome, timer, prepared.fragmentOutputs, prepared.contentKey);
    if (plan.nothing) return false;
    if (plan.generatedKey && !(prepared.recordRecipe && RuntimeSwitches::Set(RuntimeSwitch::DirectRectLists))) prepared.recipeOut = nullptr;
    inputs = prepareDrawInputs(context, state, draw, shaders, plan, timer, timing);
    if (inputs.nothing) {
        if (VerifyDrawPlan()) requireDrawPlanRepeats(context, state, draw, shaders, plan);
        return false;
    }
    timer.lap(FullDrawStep::Inputs);
    const auto* args = prepared.args = draw.indirect ? &*draw.indirect : nullptr;
    const auto indexBytes = inputs.indexBytes;
    const int dumpLimit = prepared.dumpLimit = DumpTargetLimit();
    auto& targets = prepared.targets;
    targets.resize(state.colors.size());
    auto& targetViews = prepared.targetViews;
    const bool checkedAttachments = prepared.passAttachments && CheckedPassAttachments();
    const bool verifyAttachments = prepared.passAttachments && VerifyPassAttachments() && !checkedAttachments && memoAttachments(context, state, prepared, false);
    prepared.attachmentsFromMemo = prepared.passAttachments && (!VerifyPassAttachments() || checkedAttachments) && memoAttachments(context, state, prepared, true);
    const bool memoActive = TargetMemoEnabled();
    if (!memoActive) dropTargetMemo();
    const bool memoConsulted = memoActive && !prepared.attachmentsFromMemo;
    const bool memoServes = memoConsulted && (!verifyAttachments || VerifyTargetMemo());
    bool targetsUsed = false;
    TargetProof targetProof;
    TargetProof admissionProof;
    bool repeatedColors = false;
    if (memoConsulted) {
        static_cast<void>(ownsTargetMemo(ThreadTargetMemo()));
        targetProof = currentTargetProof(context);
        admissionProof = targetProof;
    }
    if (memoServes) {
        targetsUsed = true;
        repeatedColors = repeatColors(context, state, targetProof, targets, targetViews);
    }
    const auto prepareColors = [&] {
        if (targetsUsed) {
            targets.clear();
            targets.resize(state.colors.size());
        }
        targetsUsed = true;
        targetViews.clear();
        for (std::size_t index = 0; index < state.colors.size(); ++index) {
            auto& binding = targets[index];
            binding.color = state.colors[index];
            const auto& color = binding.color;
            binding.gpuTiling = color.tileMode == ColorTileMode::RenderTarget && context.detiler != nullptr;
            if (color.samples > 1) {
                binding.multisampled = true;
                timer.phase(PhaseSetup);
                if (color.staged) {
                    const auto packed = StagedMergeTarget(color);
                    binding.stagedTemporary = AcquireStagedTemporary(context, color.extent, color.samples);
                    PrepareMultisampleTargetForDraw_nid_no_patch(context, packed);
                    static_cast<void>(MultisampleTargetView(context, packed, binding.multisampleOwner, &binding.multisampleImage));
                    const auto storage = MultisampleTargetStorageView(context, TargetMergeSurface(packed));
                    binding.stagedTarget = storage.view;
                    binding.stagedTargetOwner = storage.owner;
                    binding.stagedTargetKeep = storage.keep;
                    Require(MultisampleTargetStorable(context, packed), "a partial write of a multisampled packed color target needs a multisample image of this form to allow storage");
                    targetViews.push_back(binding.stagedTemporary->View());
                    timer.phase(PhaseReadTarget);
                    continue;
                }
                PrepareMultisampleTargetForDraw_nid_no_patch(context, color);
                targetViews.push_back(MultisampleTargetView(context, color, binding.multisampleOwner, &binding.multisampleImage));
                timer.phase(PhaseReadTarget);
                continue;
            }
            APS5_LOG_OUT_DEBUG("Creating color target %zu address=0x%llx bytes=%llu extent=%ux%u", index, static_cast<unsigned long long>(color.address), static_cast<unsigned long long>(color.bytes), color.extent.width, color.extent.height);
            const ColorTargetLayout colorLayout(color.extent.width, color.extent.height, color.tileMode, color.elementBytes);
            static_cast<void>(FlushDepthPlanes(color.address, colorLayout.Bytes(), DepthPlaneScope::Whole));
            RetireDepthSurfaces(context.device, color.address, colorLayout.Bytes());
            ClaimMultisampleTargetMemory(color.address, colorLayout.Bytes());
            if (color.fmaskAddress != 0) GuardOneFragmentFmask(context, color);
            constexpr VkBufferUsageFlags copies = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            timer.phase(PhaseSetup);
            // Debug aid: APS5_NO_RESIDENT_TARGETS=1 copies every target in and out again.
            static const bool residentTargets = std::getenv("APS5_NO_RESIDENT_TARGETS") == nullptr;
            static const bool residentLinear = std::getenv("APS5_NO_RESIDENT_LINEAR") == nullptr;
            const bool linearResident = residentLinear && residentTargets && context.detiler != nullptr && color.tileMode == ColorTileMode::Linear && (ExistingHostImportFor(context, color.address, colorLayout.Bytes()) != nullptr || (GuestMemory::Watched(color.address, colorLayout.Bytes()) && HostImportFor(context, color.address, colorLayout.Bytes(), ImportRequester::ColorTarget) != nullptr));
            if ((binding.gpuTiling || ((!ColorTileModeIsLinear(color.tileMode) || linearResident) && context.detiler != nullptr)) && residentTargets) {
                // The lookup refreshes the image on every draw (StorageTexture::Refresh: FlushPending,
                // CollectWrites over the target's pages, the DCC key scan of TextureClearKeys, then
                // UnchangedSince). The page walk is skipped while the worker's collect epoch lasts
                // (GuestMemory::BumpCollectEpoch: ordering points of the queue, or every packet under
                // APS5_PACKET_EPOCH=1); the key scan runs on every lookup regardless. Lookups of a
                // millisecond or more are counted apart; the [texture] line says how many uploaded.
                binding.resident = refreshResidentTarget(context, state, color, outcome, profile, [&] {
                    if (color.arrayLastSlice != 0) {
                        Require(color.dccAddress == 0 && color.cmaskAddress == 0 && AttachmentProxyFormat(context, color.format) == VK_FORMAT_UNDEFINED, "a slice of a 2D array color target with DCC, CMASK or an attachment proxy has no resident array image");
                        const auto array = ArraySurfaceForTarget(color);
                        // DecodeColorBuffer offsets slice s by s times this size; every slice the image stores must agree.
                        const auto sliceBytes = ComputeSurfaceSize(ComputeElementMipLayout(ColorTextureTileMode(color.tileMode), color.elementBytes, color.surfaceExtent.width, color.surfaceExtent.height, color.mipCount), 1);
                        const auto geometry = DescribeSurface(array);
                        for (std::uint32_t slice = 0; slice <= color.arrayLastSlice; ++slice) Require(geometry.GuestLayerOffset(slice) == slice * sliceBytes, "the array image's layer pitch differs from the color view's slice offset");
                        Require(color.surfaceAddress == color.arrayAddress + color.arraySlice * sliceBytes, "the array image's layer pitch differs from the color view's slice offset");
                        auto resident = CachedStorageLayers(context, array, color.arraySlice, color.layers);
                        Require(resident->Attachable(), "storage format cannot be a color attachment");
                        return resident;
                    }
                    const auto surface = SurfaceForTarget(color);
                    auto resident = RepeatStorageSurface(context, surface, shapeTarget(context, prepared.shape, index));
                    if (resident != nullptr && VerifyShapeMemo()) {
                        Require(CachedStorageSurface(context, surface) == resident, "APS5_VERIFY_SHAPE_MEMO: the shape memo's resident target differs from the storage cache's");
                        noteShapeMemoVerified(false);
                    }
                    if (resident == nullptr) resident = CachedStorageSurface(context, surface);
                    Require(resident->Attachable(), "storage format cannot be a color attachment");
                    Require(color.volume || color.mipCount > 1 || color.depth > 1 || resident->GuestBytes() == colorLayout.Bytes(), "resident image layout differs from the color layout");
                    return resident;
                });
            }
            if (binding.resident != nullptr) {
                timer.phase(PhaseReadTarget);
                binding.proxied = AttachmentProxyFormat(context, color.format) != VK_FORMAT_UNDEFINED;
                targetViews.push_back(binding.proxied ? binding.resident->AttachmentProxyView() : binding.resident->AttachmentView(color.format, color.mip, color.depthSlice, color.layers, color.arraySlice));
                continue;
            }
            materializeCmaskClear(context, color, nullptr);
            Require(!color.mipTail || binding.gpuTiling, "rendering into a packed mip tail requires the GPU tiled fallback");
            Require(color.depth == 1 && (!color.volume || (color.mipCount == 1 && color.tileMode == ColorTileMode::Linear)), "rendering into a 3D color target needs the resident image of its surface");
            Require(color.layers == 1, "layered rendering into a 2D array color target needs the resident image of its array");
            Require(color.arraySlice == 0 || !ColorTileModeIsXor(color.tileMode), "rendering into a slice of a 2D array color target with an XOR tile mode needs the resident image of its array");
            if (binding.gpuTiling) {
                binding.mip = color.mipTail ? ComputeColorTargetMipLayout(color.tileMode, color.elementBytes, color.surfaceExtent.width, color.surfaceExtent.height, color.mipCount, color.mip) : ColorTargetMip(color, colorLayout);
                binding.tiled = std::make_unique<Buffer>(context, colorLayout.Bytes(), copies);
                binding.tiledDevice = std::make_unique<DeviceBuffer>(context, colorLayout.Bytes(), copies | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                binding.linearDevice = std::make_unique<DeviceBuffer>(context, colorLayout.LinearBytes(), copies | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                binding.original.resize(colorLayout.Bytes());
                GuestMemory::Read(color.address, binding.original, colorLayout.Alignment());
                std::memcpy(binding.tiled->Bytes().data(), binding.original.data(), binding.original.size());
            } else {
                const auto x0 = static_cast<std::uint32_t>(std::max(state.scissor.offset.x, 0));
                const auto y0 = static_cast<std::uint32_t>(std::max(state.scissor.offset.y, 0));
                const auto x1 = std::min<std::uint32_t>(x0 + state.scissor.extent.width, color.extent.width);
                const auto y1 = std::min<std::uint32_t>(y0 + state.scissor.extent.height, color.extent.height);
                binding.banded = color.tileMode == ColorTileMode::Linear && color.dccAddress == 0 && x0 < x1 && y0 < y1 && (x1 - x0 < color.extent.width || y1 - y0 < color.extent.height);
                if (binding.banded) {
                    binding.band = {{static_cast<std::int32_t>(x0), static_cast<std::int32_t>(y0)}, {x1 - x0, y1 - y0}};
                    binding.bandOffset = colorLayout.Offset(x0, y0);
                    binding.bandPitch = static_cast<std::uint32_t>(colorLayout.Bytes() / (std::size_t{color.extent.height} * color.elementBytes));
                    const auto bandBytes = colorLayout.Offset(x1 - 1, y1 - 1) + color.elementBytes - binding.bandOffset;
                    binding.transfer = std::make_unique<Buffer>(context, bandBytes, copies);
                    binding.original.resize(bandBytes);
                    GuestMemory::Read(color.address + binding.bandOffset, binding.original, color.elementBytes);
                    std::memcpy(binding.transfer->Bytes().data(), binding.original.data(), bandBytes);
                } else {
                    binding.transfer = std::make_unique<Buffer>(context, colorLayout.LinearBytes(), copies);
                    ReadColorTarget(color, binding.transfer->Bytes());
                }
            }
            // A fast-cleared DCC target holds its clear value wherever the draw does not write.
            if (color.dccAddress != 0) {
                const auto keys = CurrentDccKeys(color.dccAddress, colorLayout.Bytes());
                if (IsDccClear(keys)) {
                    const auto pixels = binding.gpuTiling ? binding.tiled->Bytes() : binding.transfer->Bytes();
                    if (keys == DccKeys::ClearRegister) {
                        const auto texel = clearTexel(color, keys);
                        for (std::size_t offset = 0; offset + color.elementBytes <= pixels.size(); offset += color.elementBytes) std::memcpy(pixels.data() + offset, texel.data(), color.elementBytes);
                    } else if (!FillDccClear(color.format, keys, color.dccAlphaOnMsb, pixels)) {
                        static std::set<std::pair<std::uint64_t, int>> reported;
                        if (reported.size() < 32 && reported.insert({color.address, static_cast<int>(keys)}).second) std::fprintf(stderr, "[gpu] color target 0x%llx (VkFormat %d) has %s DCC keys; its stored texels are used\n", static_cast<unsigned long long>(color.address), static_cast<int>(color.format), DccKeysName(keys));
                        if (binding.gpuTiling) std::memcpy(pixels.data(), binding.original.data(), binding.original.size());
                        else ReadColorTarget(color, pixels);
                    }
                }
            }
            timer.phase(PhaseReadTarget);
            binding.target = std::make_unique<RenderTarget>(context, color, state.blends.at(color.exportIndex).blendEnable != 0);
            targetViews.push_back(binding.target->View());
        }
    };
    if (!repeatedColors && !prepared.attachmentsFromMemo) prepareColors();
    timer.lap(FullDrawStep::Targets);
    const auto prepareDepth = [&] {
        noteTargetMemo(TargetMemoEvent::DepthFallback);
        return DepthSurfaceViewWithStencilWriteIntent(context, *state.depth, state.depthWrite, state.stencilTest, writesStencilTarget(state), overwrittenStencil(state));
    };
    auto depthView = memoServes ? repeatDepth(context, state, repeatedColors ? &targetProof : nullptr) : VkImageView{VK_NULL_HANDLE};
    bool repeatedDepth = state.depth && depthView != VK_NULL_HANDLE;
    if (repeatedDepth) noteTargetMemo(TargetMemoEvent::DepthHit);
    if (state.depth && !repeatedDepth && !prepared.attachmentsFromMemo) depthView = prepareDepth();
    if (repeatedColors && (targetMemoOwner.load(std::memory_order_relaxed) != &ThreadTargetMemo() || !colorProofMatches(targetProof, currentTargetProof(context)) || !std::ranges::all_of(targets, [](const TargetBinding& binding) { return binding.resident->Cached(); }))) {
        noteTargetMemo(TargetMemoEvent::PostDepthInvalidated);
        repeatedColors = false;
        admissionProof = currentTargetProof(context);
        prepareColors();
        if (state.depth) depthView = prepareDepth();
        repeatedDepth = false;
    }
    if (repeatedColors) {
        for (const auto& binding : targets) {
            binding.resident->NoteProved();
            binding.resident->NoteUsed();
        }
        noteTargetMemo(TargetMemoEvent::Accept);
        if (state.depth && !repeatedDepth) noteTargetMemo(TargetMemoEvent::AcceptWithDepthMiss);
        if (!VerifyTargetMemo()) noteTargetMemo(TargetMemoEvent::AvoidedRefreshLookups, targets.size());
    }
    if ((repeatedColors || repeatedDepth) && VerifyTargetMemo()) {
        noteTargetMemo(TargetMemoEvent::VerifyChecked);
        const auto verifyProof = currentTargetProof(context);
        const auto ownProofBreaks = GuestMemory::OwnProofBreaks();
        const auto ownPendingMoves = StorageTexture::OwnPendingMoves();
        const auto ownRefreshWork = StorageTexture::OwnRefreshWork();
        const auto* owed = targetObligation(context, state, repeatedColors ? std::span<const TargetBinding>(targets) : std::span<const TargetBinding>{}, repeatedDepth);
        Require(owed == nullptr, std::string("APS5_VERIFY_TARGET_MEMO: a repeated target skipped ") + (owed != nullptr ? owed : ""));
        if (repeatedColors) {
            const auto repeatedViews = targetViews;
            std::vector<std::shared_ptr<StorageTexture>> repeatedResidents;
            for (const auto& binding : targets) repeatedResidents.push_back(binding.resident);
            prepareColors();
            Require(repeatedViews == targetViews && std::ranges::equal(repeatedResidents, targets, {}, {}, &TargetBinding::resident), "APS5_VERIFY_TARGET_MEMO: a repeated color differs from its lookup");
        }
        if (repeatedDepth) {
            const auto fullDepth = prepareDepth();
            Require(fullDepth == depthView, "APS5_VERIFY_TARGET_MEMO: a repeated depth differs from its lookup");
        }
        const auto* moved = StorageTexture::OwnRefreshWork() != ownRefreshWork ? "image content" : GuestMemory::OwnProofBreaks() != ownProofBreaks ? "proofs" : StorageTexture::OwnPendingMoves() != ownPendingMoves ? "pending" : verifyProof.MovedBefore(currentTargetProof(context));
        Require(moved == nullptr, std::string("APS5_VERIFY_TARGET_MEMO: a repeated target's lookup moved ") + (moved != nullptr ? moved : ""));
    }
    if (state.depth && !prepared.attachmentsFromMemo) targetViews.push_back(depthView);
    if (memoConsulted) {
        const bool retainedOwner = ownsTargetMemo(ThreadTargetMemo());
        const auto preparedProof = currentTargetProof(context);
        const bool unchangedDepth = (ThreadTargetMemo().depth.valid && ThreadTargetMemo().depth.proof == preparedProof &&
            (repeatedDepth || (!state.depth && !ThreadTargetMemo().depth.descriptor))) ||
            (state.depth && state.depth->samples != 1u && !ThreadTargetMemo().depth.valid);
        const bool unchanged = retainedOwner && repeatedColors && ThreadTargetMemo().colors.valid &&
            ThreadTargetMemo().colors.proof == preparedProof && unchangedDepth;
        if (admissionProof == preparedProof) {
            if (!unchanged) rememberTargets(state, targets, targetViews, preparedProof);
        } else {
            noteTargetMemo(TargetMemoEvent::Admission);
            ThreadTargetMemo().Invalidate();
        }
    }
    if (verifyAttachments) checkAttachments(state, prepared);
    timer.phase(PhasePrepare);
    timer.lap(FullDrawStep::Depth);
    const bool recordDraws = RecordDraws();
    auto* recorder = prepared.recorder = Recorder::Active();
    bool recordable = recordDraws && recorder != nullptr && dumpLimit == 0 && std::all_of(targets.begin(), targets.end(), [](const TargetBinding& binding) { return binding.resident != nullptr || binding.multisampled; });
    if (!recordable && ResourceCache::FateCensus()) noteNotKeyed(!recordDraws ? NotKeyedReason::RecordingOff : recorder == nullptr ? NotKeyedReason::NoRecorder : dumpLimit != 0 ? NotKeyedReason::DumpLimit : NotKeyedReason::TargetNotResident);
    auto& resolved = prepared.resolved;
    const bool replayOn = RuntimeSwitches::Set(RuntimeSwitch::ReplayHeapStage) || RuntimeSwitches::Set(RuntimeSwitch::XpReplay);
    const auto replayDraw = replayOn || ReplayCensusEnabled() ? heapReplayDrawReason(state, draw, shaders, inputs, targets, !stencilReferences.empty(), recordable) : ReplayReason::Count;
    resolved = resolveDrawResources(context, state, draw, shaders, snapshots, indexBytes, recordable, plan, outcome, timer, replayDraw, replayOn, prepared.view);
    if (stopAfterResolve.load(std::memory_order_relaxed) != 0) {
        const auto stop = stopAfterResolve.exchange(0, std::memory_order_relaxed);
        if (stop == 1) return false;
        if (stop == 2) throw std::runtime_error("injected failure after the draw's resources were resolved");
    }
    if (state.depth && writesDepthTarget(state)) NoteDepthSurfaceWrite(context, *state.depth);
    auto& resources = resolved.resources;
    for (std::size_t index = 0; index < targets.size(); ++index) {
        auto& binding = targets[index];
        if (binding.color.partialWriteMask == 0 || binding.multisampled || binding.resident == nullptr || binding.proxied || storesIntoResident(*resources, *binding.resident)) continue;
        const auto& color = binding.color;
        binding.resident = refreshResidentTarget(context, state, color, outcome, profile, [&] { return color.arrayLastSlice != 0 ? CachedStorageLayers(context, ArraySurfaceForTarget(color), color.arraySlice, color.layers) : CachedStorageSurface(context, SurfaceForTarget(color)); });
        if (binding.resident != nullptr) targetViews[index] = binding.resident->AttachmentView(color.format, color.mip, color.depthSlice, color.layers, color.arraySlice);
    }
    if (!stencilReferences.empty() && resources->ChecksFaults()) recordable = false;
    if (!stencilReferences.empty()) {
        const auto readImages = resources->StorageImages();
        const auto inputOverlaps = [&](std::uint64_t address, std::uint64_t bytes) {
            return std::ranges::any_of(inputs.inPlaceReads, [&](const auto& range) { return address < range.second && range.first < address + bytes; });
        };
        for (const auto& target : targets) {
            const bool imageRead = target.resident != nullptr && std::ranges::any_of(readImages, [&](const auto& image) { return image.first == target.resident->Image(); });
            Require(!imageRead && !resources->ReadsImage(target.resident.get()) && !resources->ReadsOverlap(target.color.MemoryAddress(), static_cast<std::size_t>(target.color.bytes)) && !inputOverlaps(target.color.MemoryAddress(), target.color.bytes), "stencil replay refuses attachment feedback reads");
        }
        if (state.depth) {
            const auto bytes = static_cast<std::size_t>(state.depth->extent.width) * state.depth->extent.height * 4u;
            Require(!resources->ReadsOverlap(state.depth->address, bytes) && !resources->ReadsOverlap(state.depth->stencilAddress, bytes) && !inputOverlaps(state.depth->address, bytes) && !inputOverlaps(state.depth->stencilAddress, bytes), "stencil replay refuses depth or stencil feedback reads");
            for (const auto& shader : shaders) {
                for (const auto& binding : shader.program->bindings) {
                    if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages && binding.role != ShaderRecompiler::DescriptorRole::ImageTable) continue;
                    const auto& guestWords = DrawGuestWords(shader, binding);
                    for (std::size_t offset = 0; offset + 8u <= guestWords.size(); offset += 8u) {
                        const auto words = std::span<const std::uint32_t>(guestWords).subspan(offset, 8u);
                        if (words[0] == 0u && (words[1] & 0xffu) == 0u) continue;
                        const auto texture = DecodeTextureResource(words);
                        Require(!DepthSurfaceAt(context, words, texture), "stencil replay refuses sampled depth-surface feedback");
                    }
                }
            }
        }
    }
    const auto& contentKey = resolved.contentKey;
    const bool cacheable = resolved.cacheable;
    built = resolved.built;
    // The memory-state decision of an indirect draw, made here after the resource build (which may
    // retire imports and wait for recorded work) and before anything is recorded, exactly as
    // VulkanDevice::dispatch decides for DISPATCH_INDIRECT: the GPU reads the records in place from
    // the host import unless storage-image results are pending over them (flushed, then synced), a
    // label or a copied buffer's write-back still has to land on them, or they are not imported; in
    // those cases (and with APS5_NO_GPU_INDIRECT_DRAW=1) the CPU reads them through the flush hook
    // and records plain draws with the same per-dimension rules. The import pointer stays valid:
    // refreshImports runs only under GuestMemory::GpuMutex, held here, and a retired import's
    // buffer is kept while the recorder is busy.
    auto& indirect = prepared.indirect;
    indirect.args = args;
    auto& records = prepared.records;
    if (args != nullptr) {
        static const bool gpuIndirectDraws = std::getenv("APS5_NO_GPU_INDIRECT_DRAW") == nullptr;
        const auto decide = [&](std::uint64_t address, std::size_t bytes, const HostImport*& import) {
            if (StorageTexture::FlushPending(address, bytes, nullptr, "indirect draw arguments")) {
                if (recorder != nullptr) {
                    Recorder::CountSync(2);
                    recorder->Sync();
                }
                return IndirectDrawPath::PendingImage;
            }
            if (recorder != nullptr && recorder->PendingLabelIn(address, bytes)) return IndirectDrawPath::PendingLabelOrCopy;
            const auto overlaps = [&](const auto& writer) { return writer->CopiedWritesOverlap(address, bytes); };
            if (context.copiedWriters != nullptr && std::any_of(context.copiedWriters->begin(), context.copiedWriters->end(), overlaps)) return IndirectDrawPath::PendingLabelOrCopy;
            if (std::any_of(DrawCopiedWriters()->begin(), DrawCopiedWriters()->end(), overlaps)) return IndirectDrawPath::PendingLabelOrCopy;
            if ((import = HostImportFor(context, address, bytes)) == nullptr) return IndirectDrawPath::NotImported;
            static const bool settledReads = std::getenv("APS5_NO_SETTLED_INDIRECT_READ") == nullptr;
            if (settledReads && !(prepared.recordRecipe && RuntimeSwitches::Set(RuntimeSwitch::DirectGpuIndirect)) && recorder != nullptr && recorder->PendingWriteSettled(address, bytes)) return IndirectDrawPath::Settled;
            return IndirectDrawPath::Gpu;
        };
        const auto rangeBytes = args->RangeBytes();
        if (rangeBytes == 0) {
            report(" indirect draw without records");
            return false;
        }
        Require(rangeBytes <= std::numeric_limits<std::size_t>::max(), "indirect draw record range overflow");
        if (!gpuIndirectDraws) indirect.path = IndirectDrawPath::Disabled;
        else if (inputs.listRestartPrimitive != 0) indirect.path = IndirectDrawPath::FeatureGap;
        else indirect.path = decide(args->arguments, static_cast<std::size_t>(rangeBytes), indirect.argumentImport);
        if (indirect.path == IndirectDrawPath::Gpu && args->countIndirect) {
            Require(context.drawIndirectCount, "indirect draw count without VK_KHR_draw_indirect_count must be resolved by the driver");
            const auto countPath = decide(args->countAddress, 4, indirect.countImport);
            if (countPath != IndirectDrawPath::Settled) indirect.path = countPath;
        }
        if (indirect.path != IndirectDrawPath::Gpu) {
            // The reads wait for the producing batch through the flush hook, as the driver's
            // resolve did for every indirect dispatch before its GPU path.
            const auto readStart = std::chrono::steady_clock::now();
            FinishLeaseWrites();
            const auto count = std::min(args->countIndirect ? Pm4::ReadDrawCount(*args) : args->count, args->count);
            records.reserve(count);
            for (std::uint32_t record = 0; record < count; ++record) records.push_back(Pm4::ReadDrawArguments(*args, record));
            indirect.readMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count();
            if (inputs.listRestartPrimitive != 0) CompactIndirectListRestart(context, draw, inputs, records);
            indirect.records = records;
            if (std::none_of(records.begin(), records.end(), [](const Pm4::DrawArguments& record) { return record.count != 0 && record.instances != 0; })) {
                CountIndirectDraw(indirect.path, indirect.readMs);
                report(" indirect draw with empty records");
                return false;
            }
        }
    }
    // A draw whose only completion work is a BDA fault check (the rect-list stages carry a fault
    // buffer, so every rect-list draw needed one) is recorded too, with the check as a completion
    // action like a dispatch's write-back: its own batch would first wait for everything recorded
    // before it. A draw that writes a copied buffer (at the menu stage: 0x48-byte constant buffers
    // inside an import but not at its storage offset alignment, so GuestBufferMemory copies them
    // and every guest buffer counts as writable) is recorded and then waited for at once: the
    // write-back runs before the next packet, without the separate command batch and fence a
    // synchronous draw takes. The copied write-back is listed in DrawCopiedWriters, which lets such
    // draws run without the wait; both consumers of the device's list in VulkanDevice.cpp
    // (DispatchIndirect's argument check and the fill HLE's range check, see Draw.hpp) consult this
    // list too. A draw holding an address-based build's lease (which pins guest allocations until
    // the write-back releases it) runs free like the dispatch path's: the completion releases the
    // lease, and a guest thread that needs a leased allocation syncs the recorder through the
    // registry's pin waiter (see SyncLeaseWork); APS5_SYNC_LEASE_DISPATCH=1 waits for such draws
    // at once as before. Debug aids:
    // APS5_SYNC_COMPLETION_DRAWS=1 keeps every draw with completion work synchronous;
    // APS5_NO_RECORDER_SYNC_DRAWS=1 gives the waited-for draws their own batch as before;
    // APS5_NO_RECORD_COPIED_DRAWS=1 waits for copied-write draws at once instead of listing them.
    static const bool syncCompletionDraws = std::getenv("APS5_SYNC_COMPLETION_DRAWS") != nullptr;
    static const bool recorderSyncDraws = std::getenv("APS5_NO_RECORDER_SYNC_DRAWS") == nullptr;
    static const bool recordCopiedDraws = std::getenv("APS5_NO_RECORD_COPIED_DRAWS") == nullptr;
    const bool completion = prepared.completion = resources->NeedsCompletion();
    const bool copiedWrites = completion && resources->HasCopiedWrites();
    // A lease only forces the wait when deferred release is off.
    const bool lease = resources->HoldsLease() && SyncLeaseWork();
    outcome.completion = completion;
    outcome.recorded = recordable;
    // Whether the copied write-back is listed in DrawCopiedWriters instead of waited for.
    auto& listed = prepared.listed;
    if (!recordable) {
        outcome.reason = !recordDraws || dumpLimit != 0 ? SyncDisabled : recorder == nullptr ? SyncNoRecorder : SyncNotResident;
    } else if (completion && (syncCompletionDraws || copiedWrites || lease)) {
        outcome.reason = syncCompletionDraws ? SyncDisabled : lease ? SyncLease : SyncCopiedWrites;
        if (syncCompletionDraws) outcome.recorded = false;
        else if (copiedWrites && !lease && recordCopiedDraws) listed = true;
        else if (recorderSyncDraws) outcome.waited = true;
        else outcome.recorded = false;
    }
    const bool recorded = prepared.recorded = outcome.recorded;
    // A build this recorded draw can share with later identical ones goes into the cache (a cache
    // hit is reusable by construction, so `recorded` holds for it; one with completion work is
    // never reusable).
    if (cacheable && built != nullptr && recorded && resources->Reusable() && SharedResourceCache().Insert(contentKey, resources)) templatesReplaced.fetch_add(1, std::memory_order_relaxed);
    if (cacheable && built != nullptr && (!recorded || !resources->Reusable()) && ResourceCache::FateCensus()) ResourceCache::NoteFate(contentKey, recorded ? ResourceCache::Fate::NotReusable : ResourceCache::Fate::NotRecorded, static_cast<std::uint8_t>(resources->NotReusableReason()));
    // A recorded draw renders into its resident targets in the general layout (recordDraw). Debug
    // aid: APS5_DRAW_TRANSITIONS=1 keeps the per-draw upload and download barriers, the layout
    // transitions of every target and one pass per draw, as before.
    static const bool drawTransitions = std::getenv("APS5_DRAW_TRANSITIONS") != nullptr;
    const bool lean = prepared.lean = recorded && !drawTransitions;
    if (resolved.replay && !lean) {
        resolved.replay = false;
        resources->RefreshHeapStage(shaders);
    }
    const bool multisampled = prepared.multisampled = std::any_of(targets.begin(), targets.end(), [](const TargetBinding& binding) { return binding.multisampled; });
    Require(lean || !multisampled, "multisampled draws are only rendered as recorded draws");
    for (const auto& binding : targets) {
        if (!binding.multisampled) continue;
        const auto& sources = resources->MultisampleSources();
        const auto reads = std::count(sources.begin(), sources.end(), binding.color.surfaceAddress);
        Require(reads == 0, "sampling a multisampled color target the draw renders to is not implemented");
    }
    for (const auto& binding : targets) {
        if (binding.color.partialWriteMask == 0 || binding.multisampled) continue;
        Require(lean && binding.resident != nullptr && !binding.proxied, "a partial write of a packed color target is merged into its resident image by a recorded draw only");
        Require(storesIntoResident(*resources, *binding.resident), "the partial write of a packed color target does not store into the target's resident image");
    }
    if (!lean && (!resolved.moved.empty() || resolved.resources->HasStagedCopies())) {
        if (resolved.resources->SpaceReuse()) static_cast<void>(resolved.resources->ReleaseSpace());
        resolved.resources = std::make_shared<ShaderResources>(context, shaders, state.color, static_cast<std::uint32_t>(state.colors.size()), draw.indexAddress, static_cast<std::size_t>(indexBytes), snapshots);
        resolved.moved.clear();
    }
    if (state.depth) RequireNoNativeDepthFeedback_nid_no_patch(context, *state.depth, resources->SampledTextures());
    timer.lap(FullDrawStep::Checks);
    APS5_LOG_CHARS_OUT_DEBUG("Creating Pipeline");
    // The state is copied only when a mask must change.
    auto& masked = prepared.masked;
    masked = maskedState(state, inputs.fragmentOutputs);
    if (std::any_of(targets.begin(), targets.end(), [](const TargetBinding& binding) { return binding.proxied; })) recipeOut = nullptr;
    for (std::size_t index = 0; index < targets.size(); ++index) {
        if (!targets[index].proxied) continue;
        if (!masked.has_value()) masked = state;
        masked->colors[index].format = AttachmentProxyFormat(context, state.colors[index].format);
        masked->blends.at(state.colors[index].exportIndex).colorWriteMask &= VK_COLOR_COMPONENT_R_BIT;
    }
    const State& pipelineState = masked.has_value() ? *masked : state;
    static const bool verifySetLayouts = std::getenv("APS5_VERIFY_SET_LAYOUTS") != nullptr;
    if (verifySetLayouts && context.descriptorCache != nullptr) {
        const auto described = DescribeSetLayouts(context, shaders);
        const auto built = resources->SetLayouts();
        Require(described.layouts.main == built.main && described.layouts.fragment == built.fragment && described.layouts.split == built.split && std::ranges::equal(described.mainKey, built.mainKey) && std::ranges::equal(described.fragmentKey, built.fragmentKey), "set layouts described from the shaders differ from the draw's");
    }
    auto& stencilReplay = prepared.stencilReplay;
    for (const auto reference : stencilReferences) {
        auto selected = pipelineState;
        selected.stencilReplayReference = reference;
        selected.stencilFront.reference = reference;
        selected.stencilBack.reference = reference;
        auto selectedPipeline = stencilReplay.empty() ? CachedPipeline(context, selected, inputs.VertexLayout(), *resources, shaders, lean ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, PipelineDefer::Refuse) : stencilReplay.front().second;
        Require(!selectedPipeline->SplitsFaces() && !selectedPipeline->SplitsSecondChannel(), "stencil replay refuses pipeline face or channel splits");
        stencilReplay.emplace_back(reference, std::move(selectedPipeline));
    }
    auto& pipeline = prepared.pipeline;
    const auto attachmentLayout = lean ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (!stencilReplay.empty()) {
        pipeline = stencilReplay.front().second;
    } else {
        const bool memoized = stencilReferences.empty() && std::none_of(targets.begin(), targets.end(), [](const TargetBinding& binding) { return binding.proxied || binding.multisampled; });
        const auto layouts = resources->SetLayouts();
        std::uint64_t storeHash = 0;
        if (memoized) pipeline = shapePipeline(context, prepared.shape, inputs.VertexLayout(), layouts, shaders, attachmentLayout, inputs.fragmentOutputs);
        if (pipeline != nullptr && VerifyShapeMemo()) {
            Require(CachedPipeline(context, pipelineState, inputs.VertexLayout(), layouts, shaders, attachmentLayout) == pipeline, "APS5_VERIFY_SHAPE_MEMO: the shape memo's pipeline differs from the pipeline store's");
            noteShapeMemoVerified(true);
        }
        const bool repeated = pipeline != nullptr;
        if (!repeated) pipeline = CachedPipeline(context, pipelineState, inputs.VertexLayout(), layouts, shaders, attachmentLayout, nullptr, &storeHash, PipelineDefer::Refuse);
        if (memoized) rememberShape(context, prepared.shape, targets, repeated ? nullptr : &pipeline, storeHash, inputs.VertexLayout(), layouts, shaders, attachmentLayout, inputs.fragmentOutputs);
    }
    // Resident targets' views are stable while their storage image lives, so the framebuffer is
    // reused with the pipeline; a per-draw RenderTarget gets a framebuffer of its own.
    auto& owners = prepared.owners;
    owners.reserve(targets.size());
    auto& framebufferOwners = prepared.framebufferOwners;
    framebufferOwners.reserve(targets.size());
    for (const auto& binding : targets) {
        owners.push_back(binding.resident);
        framebufferOwners.push_back(binding.multisampled ? binding.multisampleOwner : binding.resident);
    }
    Require(!pipeline->ReordersSamples() || std::ranges::none_of(state.colors, [](const ColorTarget& color) { return color.staged; }), "a partial write of a multisampled packed color target on a device that puts the guest's samples in order is not implemented");
    auto& framebuffer = prepared.framebuffer;
    framebuffer = pipeline->AcquireFramebuffer(targetViews, framebufferOwners, state.renderExtent, state.renderLayers);
    auto& reordered = prepared.reordered;
    auto& reorderPass = prepared.reorderPass;
    if (pipeline->ReordersSamples()) {
        reorderPass = SampleOrderPassFor(context, pipelineState, pipeline->HostSampleOrder());
        for (const auto& step : PlanSampleOrder(context, state, pipeline->HostSampleOrder())) {
            if (step.attachment < targets.size()) reordered.push_back(MultisampleTargetOrderTarget(context, targets[step.attachment].color));
            else reordered.push_back(DepthSurfaceOrderTarget(context, *state.depth));
            Require(reordered.back().method == step.method, "a reordered attachment's surface disagrees with the planned way of putting its samples in order");
        }
    }
    prepared.depthLocations = passDepthLocations(context, pipelineState, *pipeline);
    timer.phase(PhasePipeline);
    timer.lap(FullDrawStep::Pipeline);
    APS5_LOG_CHARS_OUT_DEBUG("Pipeline created");
    if (VerifyDrawPlan()) requireDrawPlanRepeats(context, state, draw, shaders, plan);
    return true;
}

void recordPreparedDraw(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, PreparedDraw& prepared) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static const bool traceDraws = std::getenv("APS5_TRACE_DRAWS") != nullptr;
    static AgcDriver::Mutex dumpMutex;
    static std::map<std::uint64_t, int> dumped;
    auto& timing = *prepared.timing;
    auto& timer = prepared.timer;
    auto& outcome = prepared.outcome;
    const auto* built = prepared.built;
    const auto waitedBefore = prepared.waitedBefore;
    auto& ownWaitedMs = prepared.ownWaitedMs;
    const auto report = [&](const char* suffix) { reportDrawEnd(state, timer, built, outcome, waitedBefore, ownWaitedMs, suffix); };
    auto* recipeOut = prepared.recipeOut;
    const auto& stencilReferences = prepared.stencilReferences;
    auto& inputs = prepared.inputs;
    const auto* args = prepared.args;
    const int dumpLimit = prepared.dumpLimit;
    auto& targets = prepared.targets;
    const auto& targetViews = prepared.targetViews;
    auto* recorder = prepared.recorder;
    auto& resolved = prepared.resolved;
    auto& resources = resolved.resources;
    const auto& contentKey = resolved.contentKey;
    const bool cacheable = resolved.cacheable;
    auto& indirect = prepared.indirect;
    const bool completion = prepared.completion;
    const bool listed = prepared.listed;
    const bool recorded = prepared.recorded;
    const bool lean = prepared.lean;
    const bool multisampled = prepared.multisampled;
    const auto& masked = prepared.masked;
    const auto& stencilReplay = prepared.stencilReplay;
    const auto& pipeline = prepared.pipeline;
    const auto& owners = prepared.owners;
    const auto& framebuffer = prepared.framebuffer;
    const auto& reordered = prepared.reordered;
    const auto& reorderPass = prepared.reorderPass;
    const auto& depthLocations = prepared.depthLocations;
    if (lean) {
        RecordedDraw record;
        record.recorder = recorder;
        record.resources = resources;
        record.moved = resolved.moved;
        record.replay = resolved.replay;
        if (record.replay) record.contentKey = resolved.contentKey;
        record.pipeline = pipeline;
        record.framebuffer = framebuffer;
        record.targetViews = targetViews;
        record.targets = owners;
        for (const auto& binding : targets) record.targetSlices.emplace_back(binding.color.arraySlice, binding.color.arrayLastSlice != 0 ? binding.color.layers : 1u);
        for (const auto& binding : targets) {
            if (binding.proxied) record.proxies.push_back(binding.resident);
            if (binding.multisampled) {
                record.multisampleImages.push_back(binding.multisampleImage);
                record.multisampleTargets.push_back(binding.multisampleOwner);
                if (binding.stagedTargetKeep != nullptr && binding.stagedTargetKeep != binding.multisampleOwner) record.multisampleTargets.push_back(binding.stagedTargetKeep);
            }
            if (binding.stagedTemporary != nullptr) {
                record.multisampleImages.push_back(binding.stagedTemporary->Image());
                const auto& color = binding.color;
                const auto bounds = VkRect2D{{0, 0}, color.extent};
                auto area = state.scissor;
                area.offset.x = std::max(area.offset.x, bounds.offset.x);
                area.offset.y = std::max(area.offset.y, bounds.offset.y);
                const auto right = std::min<std::int64_t>(static_cast<std::int64_t>(state.scissor.offset.x) + state.scissor.extent.width, color.extent.width);
                const auto bottom = std::min<std::int64_t>(static_cast<std::int64_t>(state.scissor.offset.y) + state.scissor.extent.height, color.extent.height);
                area.extent = {static_cast<std::uint32_t>(std::max<std::int64_t>(right - area.offset.x, 0)), static_cast<std::uint32_t>(std::max<std::int64_t>(bottom - area.offset.y, 0))};
                record.staged.push_back({binding.stagedTemporary, binding.stagedTarget, binding.stagedTargetOwner, color.stagedMask, color.samples, area});
            }
        }
        record.indirect = args != nullptr ? &indirect : nullptr;
        record.listed = listed;
        record.completion = completion;
        record.waited = outcome.waited;
        record.stencilReplay = stencilReplay;
        record.reordered = reordered;
        record.reorderPass = reorderPass;
        record.depthLocations = depthLocations;
        record.spaceKeep = &prepared.spaceKeep;
        if (auto* view = prepared.view; view != nullptr) {
            record.view = view;
            record.keeping.view = view;
            view->filled = true;
            view->resources = resources.get();
            view->pipeline = pipeline.get();
            view->framebuffer = framebuffer.get();
            view->key = contentKey;
            view->views.assign(targetViews.begin(), targetViews.end());
            view->targets.clear();
            for (const auto& owner : owners) view->targets.push_back(owner.get());
            view->passKey = DrawRenderPassKey(context, state, targetViews);
            view->moved.clear();
            view->movedSizes.clear();
            view->movedWords.clear();
            for (const auto& entry : resolved.moved) {
                view->moved.emplace_back(entry.allocation, entry.address);
                view->movedSizes.push_back(entry.size);
                view->movedWords.emplace_back(entry.words.begin(), entry.words.end());
            }
            view->pushStages = PushConstantStages(shaders);
            view->push = {};
            if (view->pushStages != 0) {
                view->push = AssemblePushConstants(shaders);
                resources->PatchPushConstants(view->push);
            }
            view->vertexInput = inputs.VertexLayout();
            view->fragmentOutputs = inputs.fragmentOutputs;
            view->shaderStages = inputs.shaderStages;
            view->hasDepth = state.depth.has_value();
            view->pushed = false;
            view->pushPipeline = nullptr;
            view->pushFramebuffer = nullptr;
            const auto* push = view->wantPush && resources->RecordLean() && stencilReferences.empty() && !UsesMeshWindow(shaders) && !pipeline->DescriptorHeap() ? resources->PushLayouts() : nullptr;
            if (push != nullptr) {
                const State& pipelineState = masked.has_value() ? *masked : state;
                auto pushPipeline = CachedPipeline(context, pipelineState, inputs.VertexLayout(), push->layouts, shaders, VK_IMAGE_LAYOUT_GENERAL);
                if (pushPipeline != nullptr) {
                    auto pushFramebuffer = pushPipeline->AcquireFramebuffer(targetViews, prepared.framebufferOwners, state.renderExtent, state.renderLayers);
                    view->pushPipeline = pushPipeline.get();
                    view->pushFramebuffer = pushFramebuffer.get();
                    view->held.push_back(std::move(pushPipeline));
                    view->held.push_back(std::move(pushFramebuffer));
                }
            }
            view->gpuWrites = resources->GpuWrites().size();
        }
        record.masked = prepared.masked.has_value() ? &*prepared.masked : nullptr;
        recordDraw(context, state, draw, shaders, inputs, record, outcome, timer, ownWaitedMs, prepared.trace);
        if (prepared.passAttachments) rememberAttachments(context, state, prepared);
        // The recipe for the caller's draw-cache entry (design_cpu_final M8, rule R3 for draws):
        // only a template the resource cache serves under this content key (reusable: no lease,
        // no copied writes, every direct region import- or mirror-served) of a direct draw, so a
        // hit's proof is the template's ProveCurrent and nothing needs completion work.
        const bool dataRecipes = prepared.recordRecipe || !RuntimeSwitches::Set(RuntimeSwitch::NoDataRecipes);
        // A multisampled draw gets a recipe with APS5_MSAA_RECORDS, unless a target is staged.
        const bool multisampleRecipe = multisampled && RuntimeSwitches::Set(RuntimeSwitch::MsaaRecords) && std::ranges::none_of(targets, [](const TargetBinding& binding) { return binding.multisampled && (binding.stagedTemporary != nullptr || binding.multisampleOwner == nullptr); });
        const bool indirectRecipe = args != nullptr && prepared.recordRecipe && RuntimeSwitches::Set(RuntimeSwitch::DirectGpuIndirect) && indirect.path == IndirectDrawPath::Gpu && !state.stages.mesh && inputs.listRestartPrimitive == 0 && inputs.expandedIndexCount == 0 && inputs.restartTable.empty();
        const bool spaceAsked = prepared.recordRecipe && resources->SpaceReuse() && RuntimeSwitches::Set(RuntimeSwitch::RecordHeavyDraws);
        const bool spaceShape = !state.stages.mesh && !state.stages.tessellationGeometry && state.tessellationCapture == nullptr && !state.quadList;
        if (recipeOut != nullptr && (DrawRecipes() || prepared.recordRecipe) && cacheable && !outcome.waited && (args == nullptr || indirectRecipe) && resources->Reusable() && (!resources->SpaceReuse() || (spaceAsked && spaceShape)) && (resolved.moved.empty() || dataRecipes) && (!multisampled || multisampleRecipe) && !pipeline->ReordersSamples()) {
            auto recipe = std::make_shared<DrawRecipe>();
            recipe->indirect = args != nullptr;
            if (multisampled) {
                for (const auto& binding : targets) recipe->multisampled.push_back(binding.multisampled ? 1u : 0u);
            }
            recipe->device = context.device;
            recipe->templateRef = resources;
            recipe->key = contentKey;
            recipe->pipeline = pipeline;
            recipe->framebuffer = framebuffer;
            for (const auto& owner : owners) recipe->targets.emplace_back(owner);
            recipe->targetViews = targetViews;
            recipe->passKey = DrawRenderPassKey(context, state, targetViews);
            recipe->vertexInput = inputs.VertexLayout();
            recipe->pushStages = PushConstantStages(shaders);
            if (recipe->pushStages != 0) {
                recipe->pushBytes = AssemblePushConstants(shaders);
                resources->PatchPushConstants(recipe->pushBytes);
            }
            recipe->masked = masked;
            recipe->fragmentOutputs = inputs.fragmentOutputs;
            recipe->shaderStages = inputs.shaderStages;
            recipe->hasDepth = state.depth.has_value();
            recipe->perDrawMoves = dataRecipes;
            recipe->spaceReuse = resources->SpaceReuse();
            if (recipe->spaceReuse) NoteRecordHeavy(RecordHeavyEvent::Built);
            *recipeOut = std::move(recipe);
        } else if (prepared.recordRecipe) {
            using enum RecipeRefusal;
            NoteRecipeRefusal(recipeOut == nullptr ? Dropped : !cacheable ? NotCacheable : outcome.waited ? Waited : args != nullptr ? Indirect : !resources->Reusable() ? NotReusable : resources->SpaceReuse() ? SpaceReuse : multisampled ? Multisampled : ReordersSamples);
            if (spaceAsked) NoteRecordHeavy(recipeOut == nullptr ? RecordHeavyEvent::RefusedDropped : !cacheable || !resources->Reusable() ? RecordHeavyEvent::RefusedNotCacheable : outcome.waited ? RecordHeavyEvent::RefusedWaited : args != nullptr && !indirectRecipe ? RecordHeavyEvent::RefusedIndirect : !spaceShape ? RecordHeavyEvent::RefusedShape : !resolved.moved.empty() && !dataRecipes ? RecordHeavyEvent::RefusedMoved : multisampled && !multisampleRecipe ? RecordHeavyEvent::RefusedMultisampled : RecordHeavyEvent::RefusedReorders);
        }
        verifyMovedInPlace(context, state, draw, shaders, inputs.indexBytes, *resources, resolved.moved);
        timing.Mark("draw_and_resource_release");
        report(outcome.waited ? (outcome.reason == SyncLease ? " recorded then waited (lease)" : " recorded then waited (copied writes)") : completion ? " recorded with completion" : " recorded");
        return;
    }
    if (prepared.recordRecipe) NoteRecipeRefusal(RecipeRefusal::NotRecorded);
    if (prepared.passAttachments) attachmentMemo().valid = false;
    const bool gpuIndirect = args != nullptr && indirect.path == IndirectDrawPath::Gpu;
    const auto touches = [&](std::uint64_t begin, std::uint64_t end) {
        const auto bytes = static_cast<std::size_t>(end - begin);
        if (resources->WritesOverlap(begin, bytes) || resources->ReadsOverlap(begin, bytes)) return true;
        if (!gpuIndirect) return false;
        return (begin < args->arguments + args->RangeBytes() && args->arguments < end) || (args->countIndirect && begin < args->countAddress + 4 && args->countAddress < end);
    };
    static_cast<void>(StorageTexture::StoreDrawnLinearTouching("draw access", [&](std::uint64_t begin, std::uint64_t end) { return touches(begin, end) || std::any_of(inputs.inPlaceReads.begin(), inputs.inPlaceReads.end(), [&](const auto& read) { return read.first < end && begin < read.second; }); }, resources->HoldsLease() || resources->UsesAddressTable() || resources->HasCopiedWrites()));
    if (UsesMeshWindow(shaders)) PrepareMeshWindow(context);
    APS5_LOG_CHARS_OUT_DEBUG("Creating CommandBatch");
    std::optional<CommandBatch> batch;
    if (!recorded) {
        // The color-target detiles below take their descriptor sets from a fresh batch.
        if (context.detiler != nullptr) context.detiler->BeginBatch();
        batch.emplace(context);
    }
    using CommandClass = Recorder::CommandClass;
    const auto countBarrier = [&](std::uint32_t count = 1) { if (recorded) Recorder::CountBarriers(CommandClass::Draw, count); };
    // A queued DCC key store over memory the draw writes or reads in place (unknown for an
    // address-based build), or over its GPU-side records, lands before it, as before a dispatch
    // (VulkanDevice::dispatch).
    if (recorded && recorder->HasQueuedKeyStores() && (resources->HoldsLease() || recorder->AnyQueuedKeyStore(touches))) recorder->FlushKeyStores();
    // A queued label store over such memory likewise (Recorder::RecordStore).
    if (recorded && recorder->HasQueuedStores() && (resources->HoldsLease() || recorder->AnyQueuedStore(touches))) recorder->FlushStores();
    const auto commands = recorded ? recorder->Commands() : batch->Handle();
    if (recorded) prepared.spaceKeep = reserveSpaceKeep(*recorder, *resources);
    std::optional<CommandState> batchRecording;
    if (!recorded) batchRecording.emplace().Begin(commands, false);
    auto& recording = recorded ? recorder->GraphicsState() : *batchRecording;
    if (recorded) recorder->PrepareSampleSlot();
    APS5_LOG_CHARS_OUT_DEBUG("CommandBatch created");
    // The draw's [gputime] class range: from its first barrier to the download barrier.
    const auto drawTiming = recorded ? recorder->BeginGpuTiming(CommandClass::Draw) : Recorder::NoTiming;
    if (recorded && Recorder::BarrierValidate()) {
        auto reads = resources->InPlaceReads();
        reads.insert(reads.end(), inputs.inPlaceReads.begin(), inputs.inPlaceReads.end());
        if (gpuIndirect) {
            reads.emplace_back(args->arguments, args->arguments + args->RangeBytes());
            if (args->countIndirect) reads.emplace_back(args->countAddress, args->countAddress + 4);
        }
        auto images = resources->StorageImages();
        for (const auto& binding : targets) {
            if (binding.resident != nullptr) images.emplace_back(binding.resident->Image(), true);
        }
        recorder->NoteAccess(CommandClass::Draw, Recorder::Access{reads, resources->GpuWrites(), images, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | inputs.shaderStages, resources->HoldsLease()});
    }
    std::unique_ptr<DeviceBuffer> scratch;
    std::shared_ptr<Buffer> argumentSnapshot;
    VkBuffer argumentBuffer = VK_NULL_HANDLE;
    VkDeviceSize argumentOffset = 0;
    bool rewritten = false;
    VkMemoryBarrier upload{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    upload.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    upload.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | inputs.shaderStages, 0, 1, &upload, 0, nullptr, 0, nullptr);
    countBarrier();
    if (state.depth) {
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        countBarrier();
    }
    APS5_LOG_CHARS_OUT_DEBUG("Upload barrier recorded");
    if (inputs.vertexInputBehindGpuWrite) {
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT);
        countBarrier();
    }
    const auto pixelLdsSubgroup = PixelFunctionLdsSubgroup(shaders);
    if (pixelLdsSubgroup != 0u) PreparePixelFunctionLds(context, commands, pixelLdsSubgroup);
    std::shared_ptr<Buffer> meshArguments;
    if (state.stages.mesh && state.tessellationCapture != nullptr) {
        recordCaptureBarrier(context, commands);
        countBarrier();
        meshArguments = state.tessellationCapture->header;
        argumentBuffer = meshArguments->Handle();
    } else if (state.stages.mesh && args != nullptr) {
        meshArguments = recordMeshArguments(context, commands, recorder, recorded, state, draw, indirect, countBarrier, inputs.restartTable);
        argumentBuffer = meshArguments->Handle();
    } else if (!inputs.restartTable.empty()) {
        meshArguments = directMeshArguments(context, draw, inputs.restartTable);
    } else if (gpuIndirect) {
        rewritten = recordIndirectArguments(context, commands, recorder, recorded, indirect, scratch, argumentSnapshot, argumentBuffer, argumentOffset, countBarrier);
    }
    for (auto& binding : targets) {
        if (binding.proxied) {
            binding.resident->RecordAttachmentProxyLoad(commands, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            countBarrier(2);
            continue;
        }
        if (binding.resident != nullptr) {
            imageBarrier(context, commands, binding.resident->Image(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, {VK_IMAGE_ASPECT_COLOR_BIT, binding.color.mip, 1, binding.resident->ImageDepth() == 1 ? binding.color.arraySlice : 0, binding.resident->ImageDepth() == 1 ? binding.color.layers : 1});
            countBarrier();
            continue;
        }
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {binding.color.extent.width, binding.color.extent.height, 1};
        if (binding.banded) {
            copy.bufferRowLength = binding.bandPitch;
            copy.imageOffset = {binding.band.offset.x, binding.band.offset.y, 0};
            copy.imageExtent = {binding.band.extent.width, binding.band.extent.height, 1};
        }
        const VkBuffer linear = binding.gpuTiling ? binding.linearDevice->Handle() : binding.transfer->Handle();
        countBarrier(binding.gpuTiling ? 4 : 2);
        if (binding.gpuTiling) {
            CopyBuffer(context, commands, binding.tiled->Handle(), 0, binding.tiledDevice->Handle(), 0, binding.original.size());
            memoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            context.detiler->Dispatch(commands, TextureTileMode::kR64KBX, binding.color.elementBytes, binding.tiledDevice->Handle(), 0, binding.linearDevice->Handle(), 0, binding.mip, false, 0, false, {.pipeBankXor = binding.color.pipeBankXor});
            memoryBarrier(context, commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        }
        imageBarrier(context, commands, binding.target->Image(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        context.Resolved(&DeviceFunctions::cmdCopyBufferToImage, "vkCmdCopyBufferToImage")(commands, linear, binding.target->Image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        PerformanceTrace::Amount("Graphics.Copy", 1);
        PerformanceTrace::Amount("Graphics.Copy.bytes", binding.original.size());
        imageBarrier(context, commands, binding.target->Image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    }
    std::vector<std::shared_ptr<void>> reorderKept;
    RecordSampleOrder(context, recording, reorderPass, reordered, true, reorderKept);
    APS5_LOG_OUT_DEBUG("Beginning pipeline renderExtent=%ux%u", state.renderExtent.width, state.renderExtent.height);
    pipeline->Begin(recording, *framebuffer, state.renderExtent, state, masked.has_value() ? &*masked : nullptr);
    APS5_LOG_CHARS_OUT_DEBUG("Pipeline Begin OK");
    const bool heapPipeline = pipeline->DescriptorHeap();
    const auto drawUse = recorded && !heapPipeline ? resources->BeginDrawUse(*recorder) : nullptr;
    if (heapPipeline) {
        resources->BindDescriptorHeaps(commands);
        NoteEmptyResourceSet();
    } else resources->BindDrawUse(commands, pipeline->Layout(), drawUse.get());
    if (pixelLdsSubgroup != 0u && !heapPipeline) BindPixelFunctionLds(context, commands, pipeline->Layout());
    if (!heapPipeline && UsesMeshWindow(shaders)) BindMeshWindow(context, commands, pipeline->Layout());
    APS5_LOG_CHARS_OUT_DEBUG("Resources bound");
    pushDrawConstants(*pipeline, commands, state, draw, shaders, *resources, nullptr, 0, meshArguments != nullptr ? meshArguments->DeviceAddress() : 0, !inputs.restartTable.empty(), inputs.meshLayerStride);
    APS5_LOG_CHARS_OUT_DEBUG("Push constants recorded");
    std::uint32_t subdraw = 0;
    if (recorded) noteDrawFaultToken(context, *recorder, commands, shaders, draw, static_cast<std::uint32_t>(state.stages.mesh.has_value() ? AgcDriver::FaultStage::mesh : AgcDriver::FaultStage::graphics), args);
    if (stencilReplay.empty()) recordDrawCommands(context, commands, state, draw, inputs, args != nullptr ? &indirect : nullptr, argumentBuffer, argumentOffset, prepared.trace, subdraw);
    else {
        recordStencilReplay(context, recording, state, draw, inputs, stencilReplay, shaders, *resources);
        if (recorded) for (const auto& [reference, selectedPipeline] : stencilReplay) recorder->Keep(selectedPipeline);
    }
    if (pipeline->SplitsFaces()) {
        pipeline->ContinueBackFaces(recording, state, masked.has_value() ? &*masked : nullptr);
        recordDrawCommands(context, commands, state, draw, inputs, args != nullptr ? &indirect : nullptr, argumentBuffer, argumentOffset, prepared.trace, subdraw);
    }
    if (pipeline->SplitsSecondChannel()) {
        pipeline->ContinueSecondChannel(recording, state, masked.has_value() ? &*masked : nullptr);
        recordDrawCommands(context, commands, state, draw, inputs, args != nullptr ? &indirect : nullptr, argumentBuffer, argumentOffset, prepared.trace, subdraw);
    }
    if (meshArguments != nullptr && recorded) recorder->Keep(meshArguments);
    if (argumentSnapshot != nullptr && recorded) recorder->Keep(argumentSnapshot);
    if (state.tessellationCapture != nullptr && recorded) {
        recorder->Keep(state.tessellationCapture->header);
        recorder->Keep(state.tessellationCapture->items);
    }
    if (args != nullptr) CountIndirectDraw(indirect.path, indirect.readMs, rewritten);
    auto checkRecords = indirectRecordCheck(args != nullptr ? &indirect : nullptr);
    APS5_LOG_CHARS_OUT_DEBUG("Draw recorded");
    if (recorded) recorder->EndPassSamples();
    EndRenderPass(context, commands);
    RecordSampleOrder(context, recording, reorderPass, reordered, false, reorderKept);
    if (depthLocations) {
        RecordDepthSampleLocations(context, commands, *depthLocations);
        countBarrier(2);
        reorderKept.push_back(depthLocations->keep);
    }
    if (recorded) {
        for (auto& object : reorderKept) recorder->Keep(std::move(object));
    }
    APS5_LOG_CHARS_OUT_DEBUG("Render pass ended");
    for (auto& binding : targets) {
        if (binding.proxied) {
            binding.resident->RecordAttachmentProxyStore(commands, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            countBarrier(2);
            continue;
        }
        if (binding.resident != nullptr) {
            imageBarrier(context, commands, binding.resident->Image(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, {VK_IMAGE_ASPECT_COLOR_BIT, binding.color.mip, 1, binding.resident->ImageDepth() == 1 ? binding.color.arraySlice : 0, binding.resident->ImageDepth() == 1 ? binding.color.layers : 1});
            countBarrier();
            continue;
        }
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {binding.color.extent.width, binding.color.extent.height, 1};
        if (binding.banded) {
            copy.bufferRowLength = binding.bandPitch;
            copy.imageOffset = {binding.band.offset.x, binding.band.offset.y, 0};
            copy.imageExtent = {binding.band.extent.width, binding.band.extent.height, 1};
        }
        countBarrier(binding.gpuTiling ? 4 : 2);
        imageBarrier(context, commands, binding.target->Image(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        const VkBuffer linear = binding.gpuTiling ? binding.linearDevice->Handle() : binding.transfer->Handle();
        VkBufferMemoryBarrier reuse{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        reuse.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        reuse.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        reuse.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        reuse.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        reuse.buffer = linear;
        reuse.size = VK_WHOLE_SIZE;
        context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &reuse, 0, nullptr);
        context.Resolved(&DeviceFunctions::cmdCopyImageToBuffer, "vkCmdCopyImageToBuffer")(commands, binding.target->Image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, linear, 1, &copy);
        PerformanceTrace::Amount("Graphics.Copy", 1);
        PerformanceTrace::Amount("Graphics.Copy.bytes", binding.original.size());
        if (binding.gpuTiling) {
            memoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
            if (dumpLimit > 0) {
                std::lock_guard lock(dumpMutex);
                if (dumped[binding.color.address] < dumpLimit) {
                    binding.dump = std::make_unique<Buffer>(context, binding.linearDevice->Size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                    CopyBuffer(context, commands, binding.linearDevice->Handle(), 0, binding.dump->Handle(), 0, binding.linearDevice->Size());
                }
            }
            context.detiler->Dispatch(commands, TextureTileMode::kR64KBX, binding.color.elementBytes, binding.linearDevice->Handle(), 0, binding.tiledDevice->Handle(), 0, binding.mip, true, 0, false, {.pipeBankXor = binding.color.pipeBankXor});
            memoryBarrier(context, commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            CopyBuffer(context, commands, binding.tiledDevice->Handle(), 0, binding.tiled->Handle(), 0, binding.original.size());
        }
    }
    VkMemoryBarrier download{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    download.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    download.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT | inputs.shaderStages, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &download, 0, nullptr, 0, nullptr);
    countBarrier();
    if (recorded) recorder->EndGpuTiming(drawTiming);
    APS5_LOG_CHARS_OUT_DEBUG("Download barrier recorded");
    APS5_LOG_CHARS_OUT_DEBUG("SubmitAndWait");
    timer.phase(PhaseRecord);
    if (recorded) {
        keepRecordedDraw(*recorder, resources, pipeline, framebuffer, inputs, owners, [&] { std::vector<std::pair<std::uint32_t, std::uint32_t>> slices; for (const auto& binding : targets) slices.emplace_back(binding.color.arraySlice, binding.color.arrayLastSlice != 0 ? binding.color.layers : 1u); return slices; }(), state.scissor, std::move(scratch), std::move(checkRecords), listed, completion, outcome, drawUse, prepared.spaceKeep != nullptr ? prepared.spaceKeep.get() : nullptr);
        timer.phase(PhaseKeep);
        if (outcome.waited) {
            // The wait the dispatch path makes for such work (source 3, "address-based"): the
            // write-back runs before the next packet, so the CPU never reads copied results or
            // touches a leased allocation before they landed.
            Recorder::CountSync(3);
            const auto ownBefore = profile ? Recorder::ThreadWaitedMs() : 0.0;
            recorder->Sync();
            if (profile) ownWaitedMs += Recorder::ThreadWaitedMs() - ownBefore;
            timer.phase(PhaseSync);
        }
        report(outcome.waited ? (outcome.reason == SyncLease ? " recorded then waited (lease)" : " recorded then waited (copied writes)") : completion ? " recorded with completion" : " recorded");
        return;
    }
    {
        // The recorder sync SubmitAndWait makes is the draw's own wait, not a nested hook wait.
        const auto ownBefore = profile ? Recorder::ThreadWaitedMs() : 0.0;
        batch->SubmitAndWait();
        if (!stencilReferences.empty()) resources->CheckFaults();
        if (profile) ownWaitedMs += Recorder::ThreadWaitedMs() - ownBefore;
    }
    if (checkRecords) checkRecords();
    timer.phase(PhaseSync);
    APS5_LOG_CHARS_OUT_DEBUG("SubmitAndWait OK");
    for (const auto& binding : targets) GuestMemory::CheckRange(reinterpret_cast<const void*>(binding.color.MemoryAddress()), binding.color.bytes, 256, true);
    for (auto& binding : targets) {
        if (!binding.dump) continue;
        std::lock_guard lock(dumpMutex);
        char name[64];
        std::snprintf(name, sizeof(name), "target_%llx_%d.raw", static_cast<unsigned long long>(binding.color.address), dumped[binding.color.address]++);
        if (std::FILE* file = std::fopen(name, "wb")) {
            const std::uint32_t header[3] = {binding.color.extent.width, binding.color.extent.height, static_cast<std::uint32_t>(binding.color.format)};
            std::fwrite(header, sizeof(header), 1, file);
            std::fwrite(binding.dump->Bytes().data(), 1, binding.dump->Bytes().size(), file);
            std::fclose(file);
        }
    }
    APS5_LOG_CHARS_OUT_DEBUG("Shader resources WriteBack");
    resources->WriteBack();
    APS5_LOG_CHARS_OUT_DEBUG("Shader resources WriteBack OK");
    static const bool skipTargetWrite = std::getenv("APS5_NO_TARGET_WRITEBACK") != nullptr;
    for (const auto& binding : targets) {
        if (skipTargetWrite) break;
        if (binding.resident != nullptr) {
            // The results stay on the GPU until something reads the target's memory.
            binding.resident->MarkSliceDirty(binding.color.arraySlice, binding.color.arrayLastSlice != 0 ? binding.color.layers : 1u);
            continue;
        }
        if (binding.gpuTiling) GuestMemory::WriteChanged(binding.color.address, binding.tiled->Bytes(), binding.original);
        else if (binding.banded) GuestMemory::WriteChanged(binding.color.address + binding.bandOffset, binding.transfer->Bytes(), binding.original);
        else WriteColorTarget(binding.color, binding.transfer->Bytes());
        // The stored texels are the whole target now, so later reads must see them rather than a fast clear.
        const auto targetBytes = ColorTargetLayout(binding.color.extent.width, binding.color.extent.height, binding.color.tileMode, binding.color.elementBytes).Bytes();
        MarkDccUncompressed(binding.color.dccAddress, targetBytes, colorKeyCount(binding.color, targetBytes));
    }
    timer.phase(PhaseWriteBack);
    if (profile && traceDraws) {
        // A synchronous draw's inputs and target sample are described as a rendering debug aid;
        // the ~0.5 ms that takes (Describe samples every bound range and reads its DCC keys) is
        // shown as its own phase and kept out of the plain profile.
        std::size_t nonzero = 0;
        std::size_t sampled = 0;
        if (!targets.empty() && targets.front().resident == nullptr) {
            const auto bytes = targets.front().gpuTiling ? targets.front().tiled->Bytes() : targets.front().transfer->Bytes();
            sampled = bytes.size() / 64;
            for (std::size_t i = 0; i < bytes.size(); i += 64) nonzero += bytes[i] != std::byte{0};
        }
        std::fprintf(stderr, "[draw]   inputs:%s\n", resources->Describe().c_str());
        if (targets.size() > 1) {
            std::string list;
            for (const auto& binding : targets) {
                char entry[64];
                std::snprintf(entry, sizeof(entry), " 0x%llx(%d,%ux%u)", static_cast<unsigned long long>(binding.color.address), static_cast<int>(binding.color.format), binding.color.extent.width, binding.color.extent.height);
                list += entry;
            }
            std::fprintf(stderr, "[draw]   targets:%s\n", list.c_str());
        }
        timer.phase(PhaseDescribe);
        char suffix[160];
        std::snprintf(suffix, sizeof(suffix), " synchronous (%s), first 0x%llx format %d (%zu of %zu sampled bytes nonzero)", SyncReasonNames[outcome.reason], static_cast<unsigned long long>(state.color.address), static_cast<int>(state.color.format), nonzero, sampled);
        report(suffix);
    } else {
        report(" synchronous");
    }
    APS5_LOG_CHARS_OUT_DEBUG("Draw finished");
}

bool IdentityCensus() {
    static const bool census = std::getenv("APS5_XP_IDENTITY_CENSUS") != nullptr;
    return census;
}

struct IdentityOutcome {
    const void* pipeline = nullptr;
    const void* resources = nullptr;
    const void* framebuffer = nullptr;
    std::uint64_t targets = 0;
    std::uint64_t inputs = 0;
};

struct IdentityEntry {
    std::uint64_t lastFrame = 0;
    std::uint64_t seen = 0;
    IdentityOutcome last;
};

enum IdentityMiss : std::size_t { IdentityMissNotRecorded, IdentityMissPipeline, IdentityMissResources, IdentityMissFramebuffer, IdentityMissTargets, IdentityMissInputs, IdentityMissBuild, IdentityMissWalk, IdentityMissCount };
constexpr std::array<const char*, IdentityMissCount> IdentityMissNames{"not recorded", "pipeline", "resources", "framebuffer", "targets", "inputs", "build", "target walk"};

struct IdentityCensusState {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, IdentityEntry> entries;
    std::uint64_t draws = 0, recipeDraws = 0, keyless = 0, matchedPrevious = 0, matchedOlder = 0, exact = 0;
    std::array<std::uint64_t, IdentityMissCount> misses{};
    std::uint64_t reportPresent = 0, prunePresent = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

IdentityCensusState& identityCensus() {
    static IdentityCensusState state;
    return state;
}

std::uint64_t mixIdentity(std::uint64_t hash, std::uint64_t value) {
    return (hash ^ value) * 0x100000001b3ull;
}

void reportIdentityCensus(IdentityCensusState& census, std::uint64_t present) {
    const auto now = std::chrono::steady_clock::now();
    if (now - census.lastReport < std::chrono::seconds(10)) return;
    const double seconds = std::chrono::duration<double>(now - census.lastReport).count();
    const auto presents = present - census.reportPresent;
    const double draws = static_cast<double>(std::max<std::uint64_t>(census.draws, 1));
    std::string misses;
    for (std::size_t i = 0; i < IdentityMissCount; ++i) {
        char text[64];
        std::snprintf(text, sizeof(text), "%s %s %.1f%%", i == 0 ? "" : ",", IdentityMissNames[i], 100.0 * static_cast<double>(census.misses[i]) / draws);
        misses += text;
    }
    std::fprintf(stderr, "[identity] %.1f s, %llu presents: %llu draws (%.0f per present), %llu recipe draws, %llu without a content key; seen in this or the previous present %.1f%%, two presents back %.1f%%, new %.1f%%; exact %.1f%% of draws; matched but not exact, by first difference (of all draws):%s; %zu identities held\n", seconds, static_cast<unsigned long long>(presents), static_cast<unsigned long long>(census.draws), presents == 0 ? 0.0 : static_cast<double>(census.draws) / static_cast<double>(presents), static_cast<unsigned long long>(census.recipeDraws), static_cast<unsigned long long>(census.keyless), 100.0 * static_cast<double>(census.matchedPrevious) / draws, 100.0 * static_cast<double>(census.matchedOlder) / draws, 100.0 * static_cast<double>(census.draws - census.matchedPrevious - census.matchedOlder) / draws, 100.0 * static_cast<double>(census.exact) / draws, misses.c_str(), census.entries.size());
    census.lastReport = now;
    census.reportPresent = present;
    census.draws = census.recipeDraws = census.keyless = census.matchedPrevious = census.matchedOlder = census.exact = 0;
    census.misses = {};
}

void noteRecipeIdentity() {
    auto& census = identityCensus();
    std::lock_guard lock(census.mutex);
    ++census.recipeDraws;
}

void noteIdentity(const PreparedDraw& prepared, const State& state, const Pm4::DrawParameters& draw) {
    std::uint64_t id = mixIdentity(0xcbf29ce484222325ull, prepared.shape);
    if (prepared.contentKey != nullptr) {
        for (const auto word : *prepared.contentKey) id = mixIdentity(id, word);
    }
    for (const auto& color : state.colors) id = mixIdentity(mixIdentity(id, color.address), static_cast<std::uint64_t>(color.format));
    if (state.depth) id = mixIdentity(mixIdentity(id, state.depth->address), state.depth->stencilAddress);
    id = mixIdentity(id, draw.indexAddress);
    id = mixIdentity(id, draw.indexCount);
    id = mixIdentity(id, draw.instanceCount);
    id = mixIdentity(id, draw.firstVertex);
    id = mixIdentity(id, draw.firstInstance);
    id = mixIdentity(id, draw.indexSize | (static_cast<std::uint64_t>(draw.indexed) << 32u));
    IdentityOutcome outcome{prepared.pipeline.get(), prepared.resolved.resources.get(), prepared.framebuffer.get(), 0, 0};
    for (const auto view : prepared.targetViews) outcome.targets = mixIdentity(outcome.targets, reinterpret_cast<std::uint64_t>(view));
    outcome.inputs = mixIdentity(mixIdentity(0, reinterpret_cast<std::uint64_t>(prepared.inputs.indexHandle)), prepared.inputs.indexOffset);
    for (std::size_t i = 0; i < prepared.inputs.vertexHandles.size(); ++i) outcome.inputs = mixIdentity(mixIdentity(outcome.inputs, reinterpret_cast<std::uint64_t>(prepared.inputs.vertexHandles[i])), i < prepared.inputs.vertexOffsets.size() ? prepared.inputs.vertexOffsets[i] : 0);
    const bool build = prepared.outcome.kind == KindBuild || prepared.outcome.kind == KindBda;
    const bool walk = prepared.outcome.pagesWalked != 0 || prepared.outcome.slowLookups != 0;
    const auto present = Recorder::Presents();
    auto& census = identityCensus();
    std::lock_guard lock(census.mutex);
    ++census.draws;
    if (prepared.contentKey == nullptr) ++census.keyless;
    auto& entry = census.entries[id];
    const auto age = present - std::min(present, entry.lastFrame);
    const bool previous = entry.seen != 0 && age <= 1;
    const bool older = entry.seen != 0 && age == 2;
    ++entry.seen;
    entry.lastFrame = present;
    auto& last = entry.last;
    if (previous) ++census.matchedPrevious;
    if (older) ++census.matchedOlder;
    if (previous || older) {
        std::optional<IdentityMiss> miss;
        if (!prepared.outcome.recorded) miss = IdentityMissNotRecorded;
        else if (outcome.pipeline != last.pipeline) miss = IdentityMissPipeline;
        else if (outcome.resources != last.resources) miss = IdentityMissResources;
        else if (outcome.framebuffer != last.framebuffer) miss = IdentityMissFramebuffer;
        else if (outcome.targets != last.targets) miss = IdentityMissTargets;
        else if (outcome.inputs != last.inputs) miss = IdentityMissInputs;
        else if (build) miss = IdentityMissBuild;
        else if (walk) miss = IdentityMissWalk;
        if (miss) ++census.misses[*miss];
        else ++census.exact;
    }
    last = outcome;
    if (present >= census.prunePresent + 32) {
        census.prunePresent = present;
        std::erase_if(census.entries, [&](const auto& item) { return item.second.lastFrame + 3 < present; });
    }
    reportIdentityCensus(census, present);
}

}

constexpr std::uint64_t MaxCapturedTriangles = 1ull << 20u;

void drawTessellationGeometry(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, const DrawTraceContext* trace) {
    using Stage = ShaderRecompiler::ShaderStage;
    Require(context.meshShader && context.geometryShader, "tessellation followed by a geometry stage needs mesh and geometry shaders");
    Require(!draw.indirect, "an indirect draw with tessellation and a geometry stage must be resolved by the driver");
    Require(draw.instanceCount <= 1u, "tessellation followed by a geometry stage is drawn for one instance only");
    if (draw.indexCount == 0 || draw.instanceCount == 0) return;
    const auto& tessellation = *state.stages.tessellation;
    const auto& geometry = *state.stages.tessellationGeometry;
    std::vector<CompiledShader> first;
    std::vector<CompiledShader> second;
    for (const auto& shader : shaders) {
        if (shader.stage != Stage::Mesh && shader.stage != Stage::Fragment) first.push_back(shader);
        if (shader.stage == Stage::Mesh || shader.stage == Stage::Fragment) second.push_back(shader);
    }
    Require(second.size() == 2u && second.front().stage == Stage::Mesh && std::ranges::count_if(first, [](const CompiledShader& shader) { return shader.stage == Stage::Geometry; }) == 1, "tessellation followed by a geometry stage needs its capture and mesh stages");
    const auto patches = static_cast<std::uint64_t>(draw.indexCount / tessellation.inputControlPoints);
    const auto level = static_cast<std::uint64_t>(std::ceil(std::clamp(state.stages.maxTessellationLevel, 1.0f, 64.0f))) + 1u;
    const auto perPatch = 2u * level * level;
    const auto capacity = static_cast<std::uint32_t>(std::min(perPatch * patches, MaxCapturedTriangles));
    auto capture = std::make_shared<TessellationCapture>();
    capture->header = std::make_shared<Buffer>(context, ShaderRecompiler::MeshCapturedHeaderBytes, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    std::array<std::uint32_t, ShaderRecompiler::MeshCapturedHeaderBytes / 4> header{};
    header[1] = 1u;
    header[2] = 1u;
    header[ShaderRecompiler::MeshCapturedCapacityDword] = capacity;
    std::memcpy(capture->header->Bytes().data(), header.data(), sizeof(header));
    capture->items = std::make_shared<Buffer>(context, static_cast<std::size_t>(capacity) * 3u * geometry.esgsItemSize * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    auto tessellated = state;
    tessellated.stages.path = ShaderPath::Tessellation;
    tessellated.stages.tessellationGeometry.reset();
    tessellated.rasterizerDiscard = true;
    tessellated.tessellationCapture = capture;
    Draw(context, tessellated, draw, first, snapshots, nullptr, trace);
    auto meshed = state;
    meshed.stages.path = ShaderPath::Geometry;
    meshed.stages.mesh = geometry;
    meshed.stages.tessellation.reset();
    meshed.stages.tessellationGeometry.reset();
    meshed.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    meshed.tessellationCapture = capture;
    const Pm4::DrawParameters captured{0, 3u, 0, 1u, 0, false, 0, 0};
    Draw(context, meshed, captured, second, snapshots, nullptr, trace);
}

std::vector<std::uint32_t> DrawResourceContentKey(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders) {
    if (!EarlyDrawKeys() || shaders.empty() || std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program == nullptr; })) return {};
    static const bool trimKey = std::getenv("APS5_NO_DRAW_KEY_TRIM") == nullptr;
    const auto indexBytes = static_cast<std::uint64_t>(draw.indexCount) * draw.indexSize;
    return DrawResourceKey(context, shaders, state.color, static_cast<std::uint32_t>(state.colors.size()), draw.indexAddress, indexBytes, !trimKey);
}

void Draw(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, std::shared_ptr<const DrawRecipe>* recipeOut, const DrawTraceContext* trace, std::uint64_t shape, const std::vector<std::uint32_t>* contentKey, const std::uint32_t* fragmentOutputs, bool recordRecipe) {
    const DepthCopySourceScope depthCopySource(state.depthCopySource);
    if (state.stages.tessellationGeometry) {
        drawTessellationGeometry(context, state, draw, shaders, snapshots, trace);
        return;
    }
    if (state.stages.mesh && state.layerExports != 0u && state.renderLayers > context.meshLimits.maxMeshOutputLayers && (draw.indirect || (draw.indexCount != 0u && draw.instanceCount != 0u))) {
        const auto limit = context.meshLimits.maxMeshOutputLayers;
        Require(limit != 0u && !state.depth, "mesh layer windows require a nonzero output limit and color-only layered rendering");
        Require(std::ranges::all_of(shaders, [](const CompiledShader& shader) { return shader.program != nullptr && shader.program->layerReplaySafe; }), "mesh layer windows refuse shaders with guest memory writes, clocks or unproven side effects");
        if (recipeOut != nullptr) recipeOut->reset();
        for (std::uint32_t base = 0; base < state.renderLayers;) {
            auto window = state;
            window.renderLayers = std::min(limit, state.renderLayers - base);
            window.meshLayerBase = base;
            window.meshLayerCount = window.renderLayers;
            for (auto& color : window.colors) {
                Require(color.layers == state.renderLayers, "mesh layer window targets have different layer counts");
                if (color.depth > 1u) color.depthSlice += base;
                else color.arraySlice += base;
                color.layers = window.renderLayers;
            }
            if (!window.colors.empty()) window.color = window.colors.front();
            std::optional<PendingPipelineScope> laterWindow;
            if (base != 0) laterWindow.emplace(false);
            Draw(context, window, draw, shaders, snapshots, nullptr, trace);
            base += window.renderLayers;
        }
        return;
    }
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    PreparedDraw prepared(profile);
    const struct SpareInputs {
        PreparedDraw& prepared;
        ~SpareInputs() { returnSpareInputs(prepared.inputs); }
    } spareInputs{prepared};
    prepared.recipeOut = recipeOut;
    prepared.trace = trace;
    prepared.shape = shape;
    prepared.contentKey = contentKey;
    prepared.fragmentOutputs = fragmentOutputs;
    prepared.passAttachments = PassAttachmentMemoEnabled();
    prepared.recordRecipe = recordRecipe;
    prepared.view = DrawCommitViewSlot::Get();
    const struct HeldSpace {
        PreparedDraw& prepared;
        ~HeldSpace() {
            const auto& resources = prepared.resolved.resources;
            if (resources == nullptr || !resources->SpaceReuse() || !resources->SpaceHeld()) return;
            if (prepared.spaceKeep != nullptr) *prepared.spaceKeep = resources->ReleaseSpace();
            else if (prepared.recorder != nullptr) resources->HoldSpaceThrough(prepared.recorder->Id(), prepared.recorder->OpenSerial());
            else static_cast<void>(resources->ReleaseSpace());
        }
    } heldSpace{prepared};
    static const bool lookupCensus = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    const auto lookupBegan = lookupCensus ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::optional<FullDrawSteps> steps;
    if (lookupCensus) prepared.steps = &steps.emplace();
    try {
        if (prepareDraw(context, state, draw, shaders, snapshots, prepared)) recordPreparedDraw(context, state, draw, shaders, prepared);
    } catch (...) {
        ThreadTargetMemo().Invalidate();
        keepHeldSpace(prepared);
        throw;
    }
    keepHeldSpace(prepared);
    if (lookupCensus) {
        const std::size_t lookup = prepared.resolved.resources == nullptr ? 2u : prepared.resolved.built != nullptr ? 1u : 0u;
        fullDrawLookups[lookup].fetch_add(1, std::memory_order_relaxed);
        fullDrawLookupNs[lookup].fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - lookupBegan).count()), std::memory_order_relaxed);
        steps->Lap(FullDrawStep::Finish);
        const auto cause = fullDrawCause();
        if (lookup == 0u) noteFullDrawSteps(cause, prepared.resolved.resources->SpaceReuse(), *steps);
        else if (lookup == 1u) fullDrawStepCensus().builds[std::min(static_cast<std::size_t>(cause), FullDrawCauseCount - 1u)].fetch_add(1, std::memory_order_relaxed);
    }
    if (IdentityCensus()) noteIdentity(prepared, state, draw);
}

DrawRecipeOutcome DrawWithRecipe(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, const DrawRecipe& recipe, const DrawTraceContext* trace) {
    const DepthCopySourceScope depthCopySource(state.depthCopySource);
    if (context.recorder != nullptr) context.recorder->BoundKeptBytes();
    static_cast<void>(snapshots);
    PerformanceTimer timing("Graphics.DrawWithRecipe");
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    DrawTimer timer(profile);
    DrawRecipeOutcome result;
    DrawOutcome outcome;
    outcome.kind = KindRecipeHit;
    if (IdentityCensus()) noteRecipeIdentity();
    const auto waitedBefore = profile ? Recorder::ThreadWaitedMs() : 0.0;
    double ownWaitedMs = 0;
    const auto miss = [&](DrawRecipeMiss reason) {
        result.miss = reason;
        if (profile) {
            auto& stats = Profile();
            std::lock_guard lock(stats.mutex);
            ++stats.recipeMisses[static_cast<std::size_t>(reason)];
        }
        return result;
    };
    Require(!draw.indirect, "a draw recipe covers direct draws only");
    auto* recorder = Recorder::Active();
    if (state.quadList || StripPatches(state, shaders) || !RecordDraws() || recorder == nullptr || DumpTargetLimit() != 0) return miss(DrawRecipeMiss::NotRecordable);
    if (recipe.hasDepth != state.depth.has_value()) return miss(DrawRecipeMiss::TargetGone);
    const std::size_t expectedViews = state.colors.size() + (recipe.hasDepth ? 1u : 0u);
    Require(recipe.targets.size() == state.colors.size() && recipe.targetViews.size() == expectedViews, "draw recipe targets do not match the state");
    const auto plan = planDraw(context, state, draw, shaders, &recipe, outcome, timer);
    DrawInputs inputs;
    if (plan.nothing) inputs.nothing = true;
    else inputs = prepareDrawInputs(context, state, draw, shaders, plan, timer, timing, &recipe);
    if (inputs.nothing) {
        result.recorded = true;
        return result;
    }
    // The resident-target proof on the stored images (design_cpu_final 3.7): while an image is
    // still the storage cache's for its surface (StorageImageCached: the cache entry, removed under
    // the cache mutex before an eviction's flush, and the LRU touch a lookup would make), it is
    // what the lookup would return, and Refresh brings it up to date exactly as the lookup's does
    // (FlushPending, the whole-surface collect and UnchangedSince, the key scan). An image gone
    // from the cache is a miss: the ordinary path looks up anew.
    auto& memo = ThreadTargetMemo();
    if (ownsTargetMemo(memo) && (memo.colors.valid || memo.depth.valid)) noteTargetMemo(TargetMemoEvent::RecipeInvalidated);
    memo.Invalidate();
    std::vector<std::shared_ptr<StorageTexture>> targets;
    targets.reserve(recipe.targets.size());
    for (std::size_t index = 0; index < recipe.targets.size(); ++index) {
        timer.phase(PhaseSetup);
        if (state.colors[index].fmaskAddress != 0) GuardOneFragmentFmask(context, state.colors[index]);
        auto stored = recipe.targets[index].lock();
        if (stored == nullptr || !StorageImageCached(context, stored.get()) || !StorageImageServesKeys(*stored, SurfaceForTarget(state.colors[index]).dccAddress)) return miss(DrawRecipeMiss::TargetGone);
        auto resident = refreshResidentTarget(context, state, state.colors[index], outcome, profile, [&] {
            if (state.colors[index].arrayLastSlice != 0) stored->RefreshLayers(state.colors[index].arraySlice, state.colors[index].layers);
            else stored->Refresh();
            return stored;
        });
        timer.phase(PhaseReadTarget);
        if (resident == nullptr) return miss(DrawRecipeMiss::TargetGone);
        targets.push_back(std::move(resident));
    }
    if (recipe.hasDepth) {
        timer.phase(PhaseSetup);
        const auto depthView = DepthSurfaceViewWithStencilWriteIntent(context, *state.depth, state.depthWrite, state.stencilTest, writesStencilTarget(state), overwrittenStencil(state));
        timer.phase(PhaseReadTarget);
        if (recipe.targetViews.back() != depthView) return miss(DrawRecipeMiss::TargetGone);
    }
    timer.phase(PhasePrepare);
    // The template's proof (rules R6/R7): ProveCurrent, T1 included, the alias checks the trimmed
    // key leaves to a hit repeated; a failure removes the template from the cache (the batch keeps
    // it) and the caller rebuilds.
    auto resources = recipe.templateRef.lock();
    if (resources == nullptr) return miss(DrawRecipeMiss::TemplateGone);
    Require(resources->Reusable(), "draw recipe over a non-reusable template");
    const auto proofStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool proved = resources->ProveCurrent(shaders, &result.proof);
    if (profile) result.proofUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - proofStart).count();
    if (!proved) {
        SharedResourceCache().Remove(recipe.key, resources.get());
        noteFailedProof(context, ProofSite::Recipe, result.proof, &recipe.key);
        countCache(&DrawProfile::cacheInvalidated);
        resources->KeepStorageImageBytes(*recorder);
        recorder->Keep(std::move(resources));
        timer.phase(PhaseLookup);
        return miss(DrawRecipeMiss::Proof);
    }
    // A data recipe's hit takes the draw's own data words and read-only buffer bases, as a
    // template hit does in resolveDrawResources; a template that cannot move them is a miss.
    std::vector<ShaderResources::MovedBuffer> moved;
    if (!recipe.perDrawMoves && std::ranges::any_of(shaders, [](const CompiledShader& shader) { return shader.heap != nullptr; })) {
        timer.phase(PhaseLookup);
        return miss(DrawRecipeMiss::Moved);
    }
    if (recipe.perDrawMoves) {
        bool patched = false;
        auto found = resources->MovedReadOnlyBuffers(shaders, *recorder, &state.color, &patched);
        if (!found.has_value()) {
            timer.phase(PhaseLookup);
            return miss(DrawRecipeMiss::Moved);
        }
        moved = std::move(*found);
    }
    if (resources->HeapStage()) {
        if (!recipe.perDrawMoves) {
            timer.phase(PhaseLookup);
            return miss(DrawRecipeMiss::Moved);
        }
        resources->RefreshHeapStage(shaders);
    }
    CheckBufferAliases(shaders, state.color, draw.indexAddress, inputs.indexBytes);
    if (recipe.hasDepth && writesDepthTarget(state)) NoteDepthSurfaceWrite(context, *state.depth);
    SharedResourceCache().Touch(recipe.key);
    countCache(&DrawProfile::cacheHits);
    timer.phase(PhaseLookup);
    outcome.completion = resources->ChecksFaults();
    outcome.recorded = true;
    // The pipeline is the recipe's while the pipeline store still holds it (its device's teardown
    // or its eviction bound drops it: a miss). The framebuffer is the recipe's while the pipeline's
    // list holds it: the views are the stored targets' (stable while they live, and they are the
    // same objects), so AcquireFramebuffer only runs when the list evicted it.
    auto pipeline = recipe.pipeline.lock();
    if (pipeline == nullptr) return miss(DrawRecipeMiss::ObjectsGone);
    auto framebuffer = recipe.framebuffer.lock();
    if (framebuffer == nullptr) {
        const std::vector<std::shared_ptr<void>> framebufferOwners(targets.begin(), targets.end());
        framebuffer = pipeline->AcquireFramebuffer(recipe.targetViews, framebufferOwners, state.renderExtent, state.renderLayers);
    }
    timer.phase(PhasePipeline);
    const auto recordStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    RecordedDraw record;
    record.recorder = recorder;
    record.masked = recipe.masked.has_value() ? &*recipe.masked : nullptr;
    record.completion = outcome.completion;
    record.resources = std::move(resources);
    record.pipeline = std::move(pipeline);
    record.framebuffer = std::move(framebuffer);
    record.targetViews = recipe.targetViews;
    record.targets = std::move(targets);
    for (const auto& color : state.colors) record.targetSlices.emplace_back(color.arraySlice, color.arrayLastSlice != 0 ? color.layers : 1u);
    record.moved = moved;
    // Push constant dwords a stage's side words replace are assembled per draw.
    const bool sidePush = std::ranges::any_of(shaders, [](const CompiledShader& shader) { return shader.heap != nullptr && !shader.heap->pushWords.empty(); });
    auto pushBytes = recipe.pushBytes;
    if (!sidePush && recipe.pushStages != 0) {
        pushBytes = AssemblePushConstants(shaders);
        record.resources->PatchPushConstants(pushBytes);
    }
    record.pushBytes = sidePush ? nullptr : &pushBytes;
    record.pushStages = recipe.pushStages;
    record.depthLocations = passDepthLocations(context, state, *record.pipeline);
    recordDraw(context, state, draw, shaders, inputs, record, outcome, timer, ownWaitedMs, trace);
    if (profile) {
        result.recordUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - recordStart).count();
        auto& stats = Profile();
        std::lock_guard lock(stats.mutex);
        ++stats.recipeHits;
    }
    verifyMovedInPlace(context, state, draw, shaders, inputs.indexBytes, *record.resources, moved);
    timing.Mark("draw_and_resource_release");
    reportDrawEnd(state, timer, nullptr, outcome, waitedBefore, ownWaitedMs, " recorded from recipe");
    result.recorded = true;
    return result;
}

namespace {

std::array<std::atomic<std::uint32_t>, static_cast<std::size_t>(DrawRecipeMiss::Count)> recordFailures{};

bool failRecordCheck(DrawRecipeMiss check) {
    auto& left = recordFailures[static_cast<std::size_t>(check)];
    auto value = left.load(std::memory_order_relaxed);
    while (value != 0 && !left.compare_exchange_weak(value, value - 1, std::memory_order_relaxed)) {}
    return value != 0;
}

}

void FailRecordChecks(DrawRecipeMiss check, std::uint32_t count) {
    recordFailures[static_cast<std::size_t>(check)].store(count, std::memory_order_relaxed);
}

namespace {
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(RecipeRefusal::Count)> recipeRefusals{};
}

void NoteRecipeRefusal(RecipeRefusal reason) {
    recipeRefusals[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

std::array<std::uint64_t, static_cast<std::size_t>(RecipeRefusal::Count)> RecipeRefusals() {
    std::array<std::uint64_t, static_cast<std::size_t>(RecipeRefusal::Count)> counts{};
    for (std::size_t index = 0; index < counts.size(); ++index) counts[index] = recipeRefusals[index].load(std::memory_order_relaxed);
    return counts;
}

bool DrawsRecordable() {
    return RecordDraws() && DumpTargetLimit() == 0;
}

namespace {

// The targets the last record draw left refreshed in the open render pass, kept for every record
// (APS5_DIRECT_PASS_TARGETS): a record that draws to the same targets under the same serials
// holds them as a record does that drew in the pass itself (DrawRecordPass). What a refresh
// depends on beside the serials is the key: the colour and depth targets as the attachment memo
// of today's draws compares them, the depth and stencil write intent, and the views.
struct SharedRecordPass {
    DrawRecordPass pass;
    std::vector<ColorTarget> colors;
    std::optional<DepthTarget> depth;
    bool depthWrite = false;
    bool stencilTest = false;
    bool stencilWritten = false;
    bool depthCopySource = false;
    std::vector<VkImageView> views;
    std::vector<const StorageTexture*> targets;
};

struct SharedRecordPassStorage {};

SharedRecordPass& sharedRecordPass() {
    return HostThreadLocal<SharedRecordPass, SharedRecordPassStorage>();
}

// APS5_XP_INPUTS_AS_IS (experiment, a ceiling only): the draw inputs of a record's last commit, taken again
// while the draw names the same index range and counts, the record holds the same recipe and no mapping was
// forgotten. Only inputs read in place are kept. A changed import, highest index or GPU write hazard is not seen.
struct InputsAsIs {
    const DrawRecipe* recipe = nullptr;
    std::uint64_t indexAddress = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t indexSize = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t firstVertex = 0;
    std::uint32_t firstInstance = 0;
    bool indexed = false;
    std::uint64_t forget = 0;
    DrawInputs inputs;
};

std::unordered_map<const DrawRecordCommit*, InputsAsIs>& inputsAsIs() {
    thread_local std::unordered_map<const DrawRecordCommit*, InputsAsIs> kept;
    return kept;
}

std::unordered_map<const DrawRecordCommit*, BindingsAsIs>& bindingsAsIs() {
    thread_local std::unordered_map<const DrawRecordCommit*, BindingsAsIs> kept;
    return kept;
}

bool takeInputsAsIs(const Context& context, const DrawRecordCommit& record, const Pm4::DrawParameters& draw, DrawInputs& inputs) {
    auto& all = inputsAsIs();
    const auto found = all.find(&record);
    if (found == all.end()) return false;
    const auto& kept = found->second;
    if (kept.recipe != record.recipe.get() || kept.indexAddress != draw.indexAddress || kept.indexCount != draw.indexCount || kept.indexSize != draw.indexSize || kept.instanceCount != draw.instanceCount || kept.firstVertex != draw.firstVertex || kept.firstInstance != draw.firstInstance || kept.indexed != draw.indexed || kept.forget != GuestMemory::ForgetSerial()) return false;
    inputs = kept.inputs;
    for (const auto& [begin, end] : inputs.inPlaceReads) context.recorder->NotePendingRead(begin, static_cast<std::size_t>(end - begin), Recorder::ReadKind::DrawInput);
    return true;
}

void keepInputsAsIs(const DrawRecordCommit& record, const Pm4::DrawParameters& draw, const DrawInputs& inputs) {
    auto& all = inputsAsIs();
    if (inputs.nothing || inputs.indices != nullptr || !inputs.restartTable.empty() || std::ranges::any_of(inputs.vertexBuffers, [](const std::shared_ptr<Buffer>& buffer) { return buffer != nullptr; })) {
        all.erase(&record);
        return;
    }
    auto& kept = all[&record];
    kept.recipe = record.recipe.get();
    kept.indexAddress = draw.indexAddress;
    kept.indexCount = draw.indexCount;
    kept.indexSize = draw.indexSize;
    kept.instanceCount = draw.instanceCount;
    kept.firstVertex = draw.firstVertex;
    kept.firstInstance = draw.firstInstance;
    kept.indexed = draw.indexed;
    kept.forget = GuestMemory::ForgetSerial();
    kept.inputs = inputs;
    kept.inputs.recycled = false;
}

bool sharedRecordPassServes(const SharedRecordPass& shared, const State& state, const DrawRecipe& recipe) {
    if (!shared.pass.valid || shared.views != recipe.targetViews || shared.colors.size() != state.colors.size() || shared.targets.size() != recipe.targets.size() || shared.depth.has_value() != state.depth.has_value() || shared.depthCopySource != state.depthCopySource) return false;
    for (std::size_t index = 0; index < state.colors.size(); ++index) {
        if (!sameColorTarget(shared.colors[index], state.colors[index])) return false;
    }
    return !state.depth || (sameDepthTarget(*shared.depth, *state.depth) && shared.depthWrite == state.depthWrite && shared.stencilTest == state.stencilTest && shared.stencilWritten == writesStencilTarget(state));
}

DrawRecordHold recordPassHold(const DrawRecordPass& pass, const Recorder& recorder, const DrawRecipe& recipe, const State& state) {
    if (!pass.valid) return DrawRecordHold::First;
    if (recipe.hasDepth && overwrittenStencil(state)) return DrawRecordHold::Depth;
    if (pass.recorder != recorder.Id() || !recorder.RenderPassOpen() || pass.generation != recorder.RenderPassGeneration()) return DrawRecordHold::Pass;
    if (pass.epoch != GuestMemory::CollectEpoch()) return DrawRecordHold::Epoch;
    if (pass.proof != GuestMemory::ProofSerial()) return DrawRecordHold::Proof;
    if (pass.pending != StorageTexture::PendingSerial()) return DrawRecordHold::Pending;
    if (pass.versions.size() != recipe.targets.size()) return DrawRecordHold::Target;
    if (recipe.hasDepth && (pass.depthSerial != DepthSurfacesSerial() || pass.depthReleased != DepthSurfacesReleased())) return DrawRecordHold::Depth;
    return DrawRecordHold::Held;
}

bool CommitBindingTimings() {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    return stats;
}

std::uint64_t commitBindingTimingOrdinal = 0;
std::uint64_t commitBindingVerifyOrdinal = 0;
std::atomic<CommitBindingsHookForTest> commitBindingsHook{nullptr};
std::atomic<ExpressHookForTest> expressHook{nullptr};

bool proveRecordTemplate(DrawRecordCommit& record, ShaderResources& resources, std::span<const CompiledShader> shaders, DrawRecipeOutcome& result, bool injected, bool* proofTrusted = nullptr) {
    if (injected) return false;
    record.effects.proved = true;
    // A ceiling for the proof work of record draws, not a mode: the template is taken as it is
    // (APS5_XP_PROOF_AS_IS), or with APS5_XP_PROOF_NONE also set, without any check at all.
    static const bool none = std::getenv("APS5_XP_PROOF_NONE") != nullptr;
    if (RuntimeSwitches::Set(RuntimeSwitch::XpCommitFloor)) return true;
    if (RuntimeSwitches::Set(RuntimeSwitch::XpProofAsIs)) return none || resources.AsIsCurrent();
    const bool trusted = RuntimeSwitches::Set(RuntimeSwitch::XpTrustCommitter);
    bool fast = trusted && RuntimeSwitches::Set(RuntimeSwitch::ExpressCommits) && resources.ExpressCurrentTrusted() == ShaderResources::ExpressMiss::None;
    bool skippedByAudit = false;
    CommitterTrustCheck check;
    if (!trusted && RuntimeSwitches::Set(RuntimeSwitch::TrustAudit) && RuntimeSwitches::Set(RuntimeSwitch::ExpressCommits) && resources.ExpressCurrentKept() == ShaderResources::ExpressMiss::None) {
        check = resources.TrustCheck(CommitterTrustFamily::Template, CommitterTrustSite::Proof);
        skippedByAudit = fast = check.Skip();
        if (!fast && resources.ExpressCurrent() != ShaderResources::ExpressMiss::None) check.Fail();
    }
    if (fast) {
        bool staged = true;
        if (auto* recorder = Recorder::Active(); recorder != nullptr) {
            try {
                resources.ExpressStage(*recorder);
            } catch (const std::exception&) {
                staged = false;
            }
        }
        if (staged) {
            result.proof = {ShaderResources::ProofPath::Fast, ShaderResources::ProofFailure::None};
            if (record.cleanProofs) record.effects.cleanProof = ShaderResources::CleanProof::Skipped;
            if (skippedByAudit && proofTrusted != nullptr) *proofTrusted = true;
            return true;
        }
    }
    if (!record.cleanProofs) {
        const bool ok = resources.ProveCurrent(shaders, &result.proof);
        if (!ok) check.Fail();
        return ok;
    }
    ShaderResources::CleanProof clean = ShaderResources::CleanProof::FirstProof;
    const bool ok = resources.ProveCurrent(shaders, &result.proof, &clean);
    record.effects.cleanProof = clean;
    if (!ok) check.Fail();
    return ok;
}

bool recordChecksQueued() {
    return std::ranges::any_of(recordFailures, [](const std::atomic<std::uint32_t>& left) { return left.load(std::memory_order_relaxed) != 0; });
}

bool repeatExcludedModes() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static const bool timeEachDraw = std::getenv("APS5_PROFILE_GPU_DRAWS") != nullptr;
    return profile || timeEachDraw || CaptureInputsEnabled() || PassAccessMemoVerify() || CaptureTrace::Enabled() || ReplayCensusEnabled() || RuntimeSwitches::Set(RuntimeSwitch::XpInputsAsIs) || RuntimeSwitches::Set(RuntimeSwitch::XpBindingsAsIs) || RuntimeSwitches::Set(RuntimeSwitch::XpCommitFloor) || RuntimeSwitches::Set(RuntimeSwitch::XpProofAsIs);
}

void fillRepeatBlock(const Context& context, const State& state, std::span<const CompiledShader> shaders, DrawRecordCommit& record, const DrawRecipe& recipe, ShaderResources& resources, const void* plainPipeline, const void* plainFramebuffer, Pipeline* pushPipeline, Framebuffer* pushFramebuffer, std::span<StorageTexture* const> residents, const ShaderResources::PushPlan* push, const ShaderResources::PreparedBuffers* bound, bool inputsPlain, std::uint64_t notesBefore, bool multisample, Recorder& recorder) {
    if (record.repeat == nullptr) record.repeat = std::make_shared<DrawRepeatBlock>();
    auto& block = *record.repeat;
    block.recipe = nullptr;
    block.detail = DrawRepeatDetail::None;
    block.sets = false;
    const auto refuse = [&](DrawRepeatClass why, DrawRepeatDetail detail = DrawRepeatDetail::None) {
        block.simple = why;
        block.detail = detail;
    };
    if (resources.HeapStage()) return refuse(DrawRepeatClass::Template, DrawRepeatDetail::HeapStage);
    if (resources.HoldsLease()) return refuse(DrawRepeatClass::Template, DrawRepeatDetail::Lease);
    if (resources.UsesAddressTable()) return refuse(DrawRepeatClass::Template, DrawRepeatDetail::AddressTable);
    if (resources.HasCopiedWrites()) return refuse(DrawRepeatClass::Template, DrawRepeatDetail::CopiedWrites);
    if (resources.ChecksFaults()) return refuse(DrawRepeatClass::Template, DrawRepeatDetail::FaultChecks);
    if (resources.SpaceReuse()) return refuse(DrawRepeatClass::Template, DrawRepeatDetail::SpaceReuse);
    if (!resources.GpuWrites().empty() || std::ranges::any_of(resources.StorageImages(), [](const std::pair<VkImage, bool>& image) { return image.second; })) return refuse(DrawRepeatClass::Template, DrawRepeatDetail::GpuWrites);
    if (resources.HasGpuCopies()) return refuse(DrawRepeatClass::Template, DrawRepeatDetail::GpuCopies);
    if (resources.HoldsDepthCopies()) return refuse(DrawRepeatClass::Template, DrawRepeatDetail::DepthCopies);
    if (!resources.RecordLean()) return refuse(DrawRepeatClass::Template, DrawRepeatDetail::NotReusable);
    const bool stencilReferences = std::ranges::any_of(shaders, [](const CompiledShader& shader) { return shader.program != nullptr && !shader.program->stencilReferences.empty(); });
    const bool functionLds = std::ranges::any_of(shaders, [](const CompiledShader& shader) { return shader.program != nullptr && shader.program->functionLdsBuffer; });
    if (shaders.size() > block.programs.size() || stencilReferences || functionLds || UsesMeshWindow(shaders) || PixelFunctionLdsSubgroup(shaders) != 0u) return refuse(DrawRepeatClass::Stages);
    if (pushPipeline == nullptr || pushFramebuffer == nullptr || push == nullptr || !record.effects.pushed) return refuse(DrawRepeatClass::NotPushed);
    if (pushPipeline->DescriptorHeap() || pushPipeline->SplitsFaces() || pushPipeline->SplitsSecondChannel()) return refuse(DrawRepeatClass::Pipeline);
    if (multisample) return refuse(DrawRepeatClass::Targets, DrawRepeatDetail::Multisampled);
    if (state.depth.has_value() && state.depth->samples != 1u) return refuse(DrawRepeatClass::Targets, DrawRepeatDetail::MultisampledDepth);
    if (residents.size() != recipe.targets.size()) return refuse(DrawRepeatClass::Targets, DrawRepeatDetail::NotRecipe);
    for (std::size_t index = 0; index < residents.size(); ++index) {
        const auto stored = recipe.targets[index].lock();
        if (residents[index] == nullptr || stored.get() != residents[index]) return refuse(DrawRepeatClass::Targets, DrawRepeatDetail::NotRecipe);
        if (residents[index]->Descriptor().tileMode == TextureTileMode::kLinear) return refuse(DrawRepeatClass::Targets, DrawRepeatDetail::Linear);
        if (state.colors[index].fmaskAddress != 0) return refuse(DrawRepeatClass::Targets, DrawRepeatDetail::Fmask);
    }
    if (bound == nullptr || record.effects.buffers != DrawRecordBuffers::Used || bound->selected.empty()) return refuse(DrawRepeatClass::Buffers);
    if ((record.effects.inputs != DrawRecordInputs::Used && record.effects.inputs != DrawRecordInputs::Rechecked) || !inputsPlain) return refuse(DrawRepeatClass::Inputs);
    if (!keepsOnceFor<StorageTexture*>(&record.keep, recorder, &resources, pushPipeline, pushFramebuffer, residents, true)) return refuse(DrawRepeatClass::Keep);
    if (!recorder.RenderPassOpen() || !record.pass.valid) return refuse(DrawRepeatClass::Pass);
    if (!passAccessMatchesFor<StorageTexture*>(record.access, &recorder, resources, residents) || record.access.readsTarget) return refuse(DrawRepeatClass::Access);
    if (repeatExcludedModes()) return refuse(DrawRepeatClass::Modes);
    if (!resources.BuildRepeatPush(*push, bound->selected, block.bindings, RuntimeSwitches::Set(RuntimeSwitch::DirectRepeatSets))) return refuse(DrawRepeatClass::Sets);
    block.sets = block.bindings.sets != nullptr;
    block.moved.clear();
    for (const auto& item : bound->selected) block.moved.push_back(item.moved);
    block.state = &state;
    block.switches = RuntimeSwitches::Generation();
    block.device = context.device;
    block.descriptorCache = context.descriptorCache;
    block.programs.fill(nullptr);
    for (std::size_t index = 0; index < shaders.size(); ++index) block.programs[index] = shaders[index].program;
    block.stages = shaders.size();
    block.resources = &resources;
    block.pipeline = pushPipeline;
    block.framebuffer = pushFramebuffer;
    block.recipePipeline = plainPipeline;
    block.recipeFramebuffer = plainFramebuffer;
    block.targets.assign(residents.begin(), residents.end());
    block.dccAddresses.clear();
    block.slices.clear();
    for (const auto& color : state.colors) {
        block.dccAddresses.push_back(SurfaceForTarget(color).dccAddress);
        block.slices.emplace_back(color.arraySlice, color.arrayLastSlice != 0 ? color.layers : 1u);
    }
    block.push = push;
    block.passKey = DrawRenderPassKey(context, state, recipe.targetViews);
    block.fenced = state.depth.has_value();
    block.writeNotes = recorder.WriteNoteCount();
    block.notesHeld = block.writeNotes == notesBefore;
    block.recipe = &recipe;
    block.simple = DrawRepeatClass::Simple;
}

bool fragmentModes() {
    static const bool timeEachDraw = std::getenv("APS5_PROFILE_GPU_DRAWS") != nullptr;
    return !timeEachDraw && !DescriptorHeapEnabled() && RecordDraws() && DumpTargetLimit() == 0 && !DrawFaultTokens() && !CaptureInputsEnabled() && !CaptureTrace::Enabled() && !ReplayCensusEnabled() && !PassAccessMemoVerify() && !GuestMemory::WriteIndexEnabled() && RuntimeSwitches::Set(RuntimeSwitch::CommitFragments) && RuntimeSwitches::Set(RuntimeSwitch::CommitBindings) && !RuntimeSwitches::Set(RuntimeSwitch::CommitBindingsDynamic) && !RuntimeSwitches::Set(RuntimeSwitch::XpBindingsAsIs) && !RuntimeSwitches::Set(RuntimeSwitch::XpCommitFloor) && !RuntimeSwitches::Set(RuntimeSwitch::XpInputsAsIs);
}

struct FragmentStreamStorage {};
struct FragmentSelectedStorage {};

CommandStream& fragmentStream() {
    auto& stream = HostThreadLocal<std::unique_ptr<CommandStream>, FragmentStreamStorage>();
    if (stream == nullptr) stream = std::make_unique<CommandStream>(CommandStream::Mode::Fragments);
    return *stream;
}

void recordFragmentDraw(const Context& context, VkCommandBuffer commands, const Pm4::DrawParameters& draw, const PreparedInputs& inputs) {
    if (!inputs.vertexHandles.empty()) context.Resolved(&DeviceFunctions::cmdBindVertexBuffers, "vkCmdBindVertexBuffers")(commands, 0, static_cast<std::uint32_t>(inputs.vertexHandles.size()), inputs.vertexHandles.data(), inputs.vertexOffsets.data());
    if (draw.indirect) {
        const auto& args = *draw.indirect;
        Require(inputs.indirectRanges != 0 && inputs.indirectRanges <= inputs.ranges.size(), "an indirect fragment needs its prepared records");
        const auto& records = inputs.ranges[inputs.ranges.size() - inputs.indirectRanges];
        if (draw.indexed) {
            const auto& range = inputs.ranges.front();
            context.Resolved(&DeviceFunctions::cmdBindIndexBuffer, "vkCmdBindIndexBuffer")(commands, range.handle, range.offset, draw.indexSize == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
        }
        if (args.countIndirect) {
            const auto& counted = inputs.ranges.back();
            CountVkDraw(VkDrawKind::Indirect);
            if (draw.indexed) context.Function<PFN_vkCmdDrawIndexedIndirectCountKHR>("vkCmdDrawIndexedIndirectCountKHR")(commands, records.handle, records.offset, counted.handle, counted.offset, args.count, args.stride);
            else context.Function<PFN_vkCmdDrawIndirectCountKHR>("vkCmdDrawIndirectCountKHR")(commands, records.handle, records.offset, counted.handle, counted.offset, args.count, args.stride);
        } else if (args.count <= 1 || context.multiDrawIndirect) {
            CountVkDraw(VkDrawKind::Indirect);
            if (draw.indexed) context.Resolved(&DeviceFunctions::cmdDrawIndexedIndirect, "vkCmdDrawIndexedIndirect")(commands, records.handle, records.offset, args.count, args.stride);
            else context.Resolved(&DeviceFunctions::cmdDrawIndirect, "vkCmdDrawIndirect")(commands, records.handle, records.offset, args.count, args.stride);
        } else {
            for (std::uint32_t record = 0; record < args.count; ++record) {
                const auto offset = records.offset + static_cast<VkDeviceSize>(record) * args.stride;
                CountVkDraw(VkDrawKind::Indirect);
                if (draw.indexed) context.Resolved(&DeviceFunctions::cmdDrawIndexedIndirect, "vkCmdDrawIndexedIndirect")(commands, records.handle, offset, 1, args.stride);
                else context.Resolved(&DeviceFunctions::cmdDrawIndirect, "vkCmdDrawIndirect")(commands, records.handle, offset, 1, args.stride);
            }
        }
        return;
    }
    if (draw.indexed) {
        const auto& range = inputs.ranges.front();
        context.Resolved(&DeviceFunctions::cmdBindIndexBuffer, "vkCmdBindIndexBuffer")(commands, range.handle, range.offset, draw.indexSize == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
        CountVkDraw(VkDrawKind::Indexed);
        context.Resolved(&DeviceFunctions::cmdDrawIndexed, "vkCmdDrawIndexed")(commands, draw.indexCount, draw.instanceCount, 0, static_cast<std::int32_t>(draw.firstVertex), draw.firstInstance);
        return;
    }
    CountVkDraw(VkDrawKind::Direct);
    context.Resolved(&DeviceFunctions::cmdDraw, "vkCmdDraw")(commands, draw.indexCount, draw.instanceCount, draw.firstVertex, draw.firstInstance);
}

std::uint64_t keptBlockVerifyOrdinal = 0;

bool keepFragmentBlock(const Context& context, DrawRecordCommit& record, const std::shared_ptr<ShaderResources>& resources, const void* pipelineKey, const void* pushKey, const ShaderResources::PushPlan* push) {
    if (!CommitFeedOn(CommitFeedBlocks) || record.fragmentsRefused || resources == nullptr || !resources->FragmentEligible() || !fragmentModes()) return false;
    if (record.fragmentResources != resources.get() || record.fragmentPipeline != pipelineKey || record.fragmentPushPipeline != pushKey) return false;
    const auto block = record.fragmentBlock.load(std::memory_order_acquire);
    if (block == nullptr || block->device != context.device || block->push != push) return false;
    if (block->resources.expired() || block->pipeline.expired() || (pushKey != nullptr && block->pushPipeline.expired())) return false;
    if (const auto every = CommitFeedVerifyEvery(); every != 0 && ++keptBlockVerifyOrdinal % every == 0) {
        const void* blockResources = block->resources.lock().get();
        const void* blockPipeline = block->pipeline.lock().get();
        const void* blockPushPipeline = block->pushPipeline.lock().get();
        const char* field = blockResources != static_cast<const void*>(resources.get()) ? "its template" : blockPipeline != pipelineKey ? "its pipeline" : blockPushPipeline != pushKey ? "its push pipeline" : nullptr;
        NoteCommitFeedVerify(field == nullptr, "kept block", field == nullptr ? std::string() : std::string(field) + " is not the object of the commit without prepared buffers; the block is withdrawn");
        if (field != nullptr) return false;
        record.feedVerify = true;
    }
    if (CommitFeedCensus()) NoteCommitFeed(CommitFeedCount::BlocksKept);
    return true;
}

void publishFragmentBlock(const Context& context, DrawRecordCommit& record, const std::shared_ptr<ShaderResources>& resources, const std::weak_ptr<Pipeline>& pipeline, const std::weak_ptr<Pipeline>& pushPipeline, const void* pipelineKey, const void* pushKey, const ShaderResources::PushPlan* push, bool eligible, bool withoutBuffers) {
    const bool wanted = eligible && !record.fragmentsRefused && resources != nullptr && resources->FragmentEligible() && fragmentModes();
    if (!wanted) {
        if (record.fragmentResources != nullptr) {
            if (withoutBuffers && keepFragmentBlock(context, record, resources, pipelineKey, pushKey, push)) return;
            record.fragmentBlock.store(nullptr);
            record.fragmentResources = nullptr;
            record.fragmentPipeline = nullptr;
            record.fragmentPushPipeline = nullptr;
            if (CommitFeedCensus()) NoteCommitFeed(CommitFeedCount::BlocksWithdrawn);
        }
        return;
    }
    if (record.fragmentResources == resources.get() && record.fragmentPipeline == pipelineKey && record.fragmentPushPipeline == pushKey) return;
    auto block = std::make_shared<DrawFragmentBlock>();
    block->resources = resources;
    block->pipeline = pipeline;
    block->pushPipeline = pushPipeline;
    block->push = push;
    block->device = context.device;
    record.fragmentBlock.store(std::shared_ptr<const DrawFragmentBlock>(std::move(block)));
    record.fragmentResources = resources.get();
    record.fragmentPipeline = pipelineKey;
    record.fragmentPushPipeline = pushKey;
}

std::uint64_t commitFragmentVerifyOrdinal = 0;

void captureExpress(const Context& context, const State& state, const DrawRecipe& recipe, DrawRecordCommit& record, const Recorder& recorder, const std::shared_ptr<ShaderResources>& resources, const DrawFragment& fragment, std::shared_ptr<Framebuffer> framebuffer, std::vector<std::shared_ptr<StorageTexture>> targets, std::vector<std::pair<std::uint32_t, std::uint32_t>> slices, std::vector<std::shared_ptr<void>> multisample, std::vector<VkImage> multisampleImages, bool proofTrusted = false) {
    record.express.reset();
    auto& noted = record.effects.fragment.expressCapture;
    if (!record.effects.fragment.spliced && !record.effects.fragment.replayed) {
        noted = DrawExpressCapture::NotSpliced;
        return;
    }
    if (resources == nullptr || framebuffer == nullptr || fragment.pipeline == nullptr) {
        noted = DrawExpressCapture::Objects;
        return;
    }
    if (WriteValidation::Session() != 0) WriteValidation::Note(WriteValidation::Test::TemplateCurrent, WriteValidation::Outcome::Kept);
    if (const auto why = RuntimeSwitches::Set(RuntimeSwitch::XpTrustCommitter) ? resources->ExpressCurrentTrusted() : proofTrusted ? resources->ExpressCurrentKept() : resources->ExpressCurrent(); why != ShaderResources::ExpressMiss::None && (why == ShaderResources::ExpressMiss::Regions || !RuntimeSwitches::Set(RuntimeSwitch::ExpressKeepState) || !resources->ExpressWatchable())) {
        noted = static_cast<DrawExpressCapture>(static_cast<std::size_t>(DrawExpressCapture::NotProved) + static_cast<std::size_t>(why) - static_cast<std::size_t>(ShaderResources::ExpressMiss::NotProved));
        return;
    }
    const auto& access = record.access;
    multisample.resize(targets.size());
    multisampleImages.resize(targets.size(), VK_NULL_HANDLE);
    bool targetsKnown = true;
    for (std::size_t index = 0; index < targets.size(); ++index) {
        if (targets[index] == nullptr && (multisample[index] == nullptr || multisampleImages[index] == VK_NULL_HANDLE)) targetsKnown = false;
    }
    if (access.resources != resources.get() || access.readsTarget || targets.size() != slices.size() || !targetsKnown) {
        noted = DrawExpressCapture::Access;
        return;
    }
    auto express = std::make_shared<DrawExpress>();
    express->recipe = &recipe;
    express->state = &state;
    express->switches = RuntimeSwitches::Generation();
    if ((RuntimeSwitches::Set(RuntimeSwitch::XpDepthLayouts) || RuntimeSwitches::Set(RuntimeSwitch::XpTargetRestarts)) && recipe.hasDepth && state.depth.has_value() && !recipe.targetViews.empty()) {
        const auto attached = DepthSurfaceAttachmentImage(context, recipe.targetViews.back()).image;
        if (samplesImage(*resources, attached)) {
            noted = DrawExpressCapture::Access;
            Recorder::CountGpuSide(Recorder::GpuSide::DepthSampleRefusals);
            return;
        }
        if (RuntimeSwitches::Set(RuntimeSwitch::XpTargetRestarts) && !GpuPathEnabled()) express->restartDepth = attached;
    }
    express->device = context.device;
    auto keep = std::make_shared<DrawExpressKeep>();
    keep->resources = resources;
    keep->pipeline = fragment.pipeline;
    keep->framebuffer = std::move(framebuffer);
    keep->targets = std::move(targets);
    keep->multisample = std::move(multisample);
    express->multisampleImages = std::move(multisampleImages);
    for (std::size_t index = 0; index < keep->targets.size(); ++index) {
        const auto* target = keep->targets[index].get();
        express->targetEpochs.push_back(target != nullptr ? target->StateEpoch() : 0u);
        express->targetVersions.push_back(target != nullptr ? target->UploadVersion() : 0u);
        express->dccAddresses.push_back(index < state.colors.size() ? SurfaceForTarget(state.colors[index]).dccAddress : 0u);
    }
    express->slices = std::move(slices);
    express->passKey = DrawRenderPassKey(context, state, recipe.targetViews);
    express->fenced = state.depth.has_value();
    express->hasDepth = recipe.hasDepth;
    express->depthSerial = DepthSurfacesSerial();
    express->depthReleased = DepthSurfacesReleased();
    express->reads = access.reads;
    express->writes = access.writes;
    express->writeOnly = access.writeOnly;
    express->images = access.images;
    express->keptRecorder = recorder.Id();
    express->keptBatch = recorder.OpenSerial();
    express->keep = std::move(keep);
    express->valid = true;
    record.express = std::move(express);
    noted = DrawExpressCapture::Captured;
}

bool gpuIndirectRecord(const Context& context, const Recorder& recorder, const Pm4::DrawParameters::IndirectDraw& args, IndirectRecord& out) {
    const auto rangeBytes = args.RangeBytes();
    if (rangeBytes == 0 || rangeBytes > std::numeric_limits<std::size_t>::max()) return false;
    const auto clear = [&](std::uint64_t address, std::size_t bytes, const HostImport*& import) {
        if (PendingStorageOverlaps(address, bytes, nullptr) || recorder.PendingLabelIn(address, bytes)) return false;
        const auto overlaps = [&](const auto& writer) { return writer != nullptr && writer->CopiedWritesOverlap(address, bytes); };
        if (context.copiedWriters != nullptr && std::any_of(context.copiedWriters->begin(), context.copiedWriters->end(), overlaps)) return false;
        const auto drawWriters = DrawCopiedWriters();
        if (std::any_of(drawWriters->begin(), drawWriters->end(), overlaps)) return false;
        import = HostImportFor(context, address, bytes);
        return import != nullptr;
    };
    out = {};
    out.args = &args;
    if (!clear(args.arguments, static_cast<std::size_t>(rangeBytes), out.argumentImport)) return false;
    if (args.countIndirect && (!context.drawIndirectCount || !clear(args.countAddress, 4u, out.countImport))) return false;
    out.path = IndirectDrawPath::Gpu;
    return true;
}

}

std::uint64_t CommitFragmentsVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_COMMIT_FRAGMENTS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

bool RecordDrawFragment(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, const DrawFragmentBlock& block, const ShaderResources::PreparedBuffers& prepared, const PreparedInputs& inputs, DrawFragment& out, DrawFragmentRefusal& refusal) {
    const auto refuse = [&](DrawFragmentRefusal why) {
        refusal = why;
        return false;
    };
    refusal = DrawFragmentRefusal::None;
    if (block.device != context.device || prepared.placedDevice != context.device || prepared.placed == nullptr || inputs.device != context.device) return refuse(DrawFragmentRefusal::Device);
    if (!fragmentModes()) return refuse(DrawFragmentRefusal::Modes);
    using IndirectRule = Pm4::DrawParameters::IndirectDraw::Rule;
    const bool indirectFragment = draw.indirect && RuntimeSwitches::Set(RuntimeSwitch::ExpressGpuIndirect) && RuntimeSwitches::Set(RuntimeSwitch::IndirectInPass) && draw.indirect->vertexRule == IndirectRule::InPlace && draw.indirect->instanceRule == IndirectRule::InPlace && inputs.indirectRanges == (draw.indirect->countIndirect ? 2u : 1u) && inputs.ranges.size() >= inputs.indirectRanges + (draw.indexed ? 1u : 0u) && (!draw.indirect->countIndirect || context.drawIndirectCount);
    if ((draw.indirect && !indirectFragment) || state.quadList || StripPatches(state, shaders) || state.stages.mesh || state.stages.tessellation || state.stages.tessellationGeometry || state.tessellationCapture != nullptr || (draw.indexed && inputs.ranges.empty())) return refuse(DrawFragmentRefusal::Stages);
    if (UsesMeshWindow(shaders) || std::ranges::any_of(shaders, [](const CompiledShader& shader) { return shader.program != nullptr && (!shader.program->stencilReferences.empty() || shader.program->functionLdsBuffer); })) return refuse(DrawFragmentRefusal::Stages);
    const auto resources = block.resources.lock();
    const auto pipeline = block.pipeline.lock();
    if (resources == nullptr || pipeline == nullptr) return refuse(DrawFragmentRefusal::Gone);
    const auto pushPipeline = block.push != nullptr ? block.pushPipeline.lock() : std::shared_ptr<Pipeline>{};
    const auto* push = pushPipeline != nullptr ? block.push : nullptr;
    const auto splits = [](const Pipeline& candidate) { return candidate.DescriptorHeap() || candidate.SplitsFaces() || candidate.SplitsSecondChannel(); };
    if (splits(*pipeline) || (pushPipeline != nullptr && splits(*pushPipeline))) return refuse(DrawFragmentRefusal::Pipeline);
    if (!resources->FragmentEligible()) return refuse(DrawFragmentRefusal::Stages);
    auto& selected = HostThreadLocal<std::vector<std::size_t>, FragmentSelectedStorage>();
    selected.clear();
    for (const auto& item : prepared.selected) selected.push_back(item.allocation);
    const bool ownSets = selected.empty() && RuntimeSwitches::Set(RuntimeSwitch::FragmentOwnSets);
    if (selected.empty() && !ownSets) return refuse(DrawFragmentRefusal::Selection);
    const bool pushing = push != nullptr && std::ranges::any_of(selected, [&](std::size_t index) { return index < push->allocationSets.size() && push->allocationSets[index] == push->set; });
    if (prepared.inPlace && (!pushing || std::ranges::any_of(prepared.selected, [&](const ShaderResources::PreparedBuffers::Selected& item) { return item.inPlace && (item.allocation >= push->allocationSets.size() || push->allocationSets[item.allocation] != push->set); }))) return refuse(DrawFragmentRefusal::Selection);
    const auto& bound = pushing ? *pushPipeline : *pipeline;
    auto bindings = std::move(prepared.placed);
    auto& stream = fragmentStream();
    const auto faultBefore = resources->ChecksFaults() ? resources->FaultGeneration() : 0u;
    try {
        if (ownSets) {
            bindings->pushSelected.clear();
        } else if (pushing) {
            const auto pushedSet = static_cast<std::int32_t>(push->set);
            bindings->pushedSet = pushedSet;
            const auto& sets = push->allocationSets;
            if (std::ranges::any_of(selected, [&](std::size_t index) { return index >= sets.size() || sets[index] != static_cast<std::uint8_t>(pushedSet); })) resources->WriteFragmentSets(*bindings, selected, pushedSet);
            else bindings->cache = resources->FragmentDescriptorCache();
        } else {
            bindings->pushSelected.clear();
            resources->WriteFragmentSets(*bindings, selected, -1);
        }
        if (resources->ChecksFaults()) {
            const auto faultAfter = resources->FaultGeneration();
            out.faultGeneration = (faultBefore & 1u) == 0 && faultAfter == faultBefore ? faultBefore : ~std::uint64_t{0};
        }
        stream.BeginFragment();
        const auto commands = stream.Handle();
        CommandState recording;
        recording.Begin(commands, false);
        bound.Continue(recording, state, inputs.recipe != nullptr && inputs.recipe->masked.has_value() ? &*inputs.recipe->masked : nullptr);
        if (ownSets) {
            resources->BindFragmentOwnSets(commands, bound.Layout());
        } else if (pushing) {
            resources->PushFragmentBindings(commands, bound.Layout(), *bindings, *push);
        } else {
            const auto set = bindings->allocation.set;
            const auto fragmentSet = bindings->fragmentAllocation.set;
            if (set != VK_NULL_HANDLE) CmdBindDescriptorSets(context, commands, VK_PIPELINE_BIND_POINT_GRAPHICS, bound.Layout(), 0, 1, &set, SetBindSite::Draw, bindings->DynamicCount(0), bindings->DynamicOffsets(0));
            if (fragmentSet != VK_NULL_HANDLE) CmdBindDescriptorSets(context, commands, VK_PIPELINE_BIND_POINT_GRAPHICS, bound.Layout(), ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet, 1, &fragmentSet, SetBindSite::Draw, bindings->DynamicCount(1), bindings->DynamicOffsets(1));
        }
        pushDrawConstants(bound, commands, state, draw, shaders, *resources, nullptr, 0, 0, false, 0, ownSets ? nullptr : bindings.get());
        recordFragmentDraw(context, commands, draw, inputs);
    } catch (const std::exception&) {
        static_cast<void>(stream.EndFragment());
        return refuse(DrawFragmentRefusal::Thrown);
    }
    out.commands = stream.EndFragment();
    out.written = bindings->allocation.set != VK_NULL_HANDLE || bindings->fragmentAllocation.set != VK_NULL_HANDLE;
    out.bindings = std::move(bindings);
    out.resources = resources;
    out.pipeline = pushing ? std::shared_ptr<const Pipeline>(pushPipeline) : std::shared_ptr<const Pipeline>(pipeline);
    out.push = push;
    out.pushing = pushing;
    out.device = context.device;
    out.indirect = draw.indirect.has_value();
    if (out.indirect) {
        const auto& records = inputs.ranges[inputs.ranges.size() - inputs.indirectRanges];
        out.arguments = records.handle;
        out.argumentOffset = records.offset;
        if (draw.indirect->countIndirect) {
            out.count = inputs.ranges.back().handle;
            out.countOffset = inputs.ranges.back().offset;
        }
    }
    return true;
}

namespace {

enum class RecordSpaceOutcome : std::uint8_t { Served, InUse, Changed, Partial, SnapshotChanged, Outside };

RecordSpaceOutcome takeRecordSpace(ShaderResources& resources, std::span<const GuestMemorySnapshot> snapshots) {
    if (auto* active = Recorder::Active(); active != nullptr) {
        GuestBufferMemory::ReleaseRetainedSpaces(active->Id(), active->CompletedThrough());
        static_cast<void>(resources.ReleaseCompletedSpace(active->Id(), active->CompletedThrough()));
    }
    if (resources.SpaceHeld()) return RecordSpaceOutcome::InUse;
    if (!resources.RetakeSpace()) return RecordSpaceOutcome::Changed;
    if (!resources.AddressTableGuardCurrent()) {
        static_cast<void>(resources.ReleaseSpace());
        return RecordSpaceOutcome::Changed;
    }
    completeSnapshotWrites(snapshots);
    const auto service = resources.SnapshotsServedInPlace(snapshots);
    if (service == SnapshotService::Served) return RecordSpaceOutcome::Served;
    static_cast<void>(resources.ReleaseSpace());
    return service == SnapshotService::Partial ? RecordSpaceOutcome::Partial : service == SnapshotService::Changed ? RecordSpaceOutcome::SnapshotChanged : RecordSpaceOutcome::Outside;
}

RecordHeavyEvent recordSpaceEvent(RecordSpaceOutcome outcome) {
    return outcome == RecordSpaceOutcome::InUse ? RecordHeavyEvent::SpaceInUse : outcome == RecordSpaceOutcome::Changed ? RecordHeavyEvent::SpaceChanged : outcome == RecordSpaceOutcome::Partial ? RecordHeavyEvent::SpacePartial : outcome == RecordSpaceOutcome::SnapshotChanged ? RecordHeavyEvent::SpaceSnapshotChanged : RecordHeavyEvent::SpaceOutside;
}

struct RecordSpaceHold {
    std::span<const GuestMemorySnapshot> snapshots;
    std::shared_ptr<ShaderResources> resources;
    Recorder* recorder = nullptr;
    std::shared_ptr<std::shared_ptr<const void>> keep;
    bool settled = false;
    int exceptions = std::uncaught_exceptions();
    RecordSpaceHold() = default;
    RecordSpaceHold(const RecordSpaceHold&) = delete;
    RecordSpaceHold& operator=(const RecordSpaceHold&) = delete;
    void KeepOnThrow() {
        if (!settled || keep != nullptr || recorder == nullptr || resources == nullptr || !resources->SpaceHeld()) return;
        keep = reserveSpaceKeep(*recorder, *resources);
    }
    ~RecordSpaceHold() {
        if (resources == nullptr || !resources->SpaceHeld()) return;
        if (!settled) static_cast<void>(resources->ReleaseSpace());
        else if (keep != nullptr) *keep = resources->ReleaseSpace();
        else if (recorder != nullptr && std::uncaught_exceptions() > exceptions) resources->HoldSpaceThrough(recorder->Id(), recorder->OpenSerial());
        else static_cast<void>(resources->ReleaseSpace());
    }
};

DrawRecipeOutcome drawWithRecord(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, DrawRecordCommit& record, const DrawTraceContext* trace, DrawCommitView* answer, const ShaderResources::PreparedBuffers* prepared, PreparedInputs* preparedInputs, DrawFragment* fragment, RecordSpaceHold& space) {
    const DepthCopySourceScope depthCopySource(state.depthCopySource);
    auto* recorder = Recorder::Active();
    const bool recording = answer == nullptr;
    const auto failed = [&](DrawRecipeMiss check) { return recording && failRecordCheck(check); };
    if (recorder != nullptr && recording) recorder->BoundKeptBytes();
    const bool repeatBlocks = recording && recorder != nullptr && RuntimeSwitches::Set(RuntimeSwitch::DirectRepeat);
    const auto notesBefore = repeatBlocks ? recorder->WriteNoteCount() : 0u;
    if (repeatBlocks && record.repeat != nullptr) record.repeat->simple = DrawRepeatClass::NotRecorded;
    PerformanceTimer timing("Graphics.DrawWithRecord");
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    DrawTimer timer(profile);
    DrawRecipeOutcome result;
    DrawOutcome outcome;
    outcome.kind = KindRecipeHit;
    record.effects = {};
    const auto waitedBefore = profile ? Recorder::ThreadWaitedMs() : 0.0;
    double ownWaitedMs = 0;
    const auto miss = [&](DrawRecipeMiss reason) {
        result.miss = reason;
        return result;
    };
    if (record.recipe == nullptr) return miss(DrawRecipeMiss::TemplateGone);
    const auto& recipe = *record.recipe;
    if ((draw.indirect && (!RuntimeSwitches::Set(RuntimeSwitch::DirectGpuIndirect) || !recipe.indirect)) || (!draw.indirect && recipe.indirect) || state.quadList || StripPatches(state, shaders) || !RecordDraws() || recorder == nullptr || DumpTargetLimit() != 0 || state.stages.mesh || state.stages.tessellationGeometry || state.tessellationCapture != nullptr) return miss(DrawRecipeMiss::NotRecordable);
    if (recipe.device != context.device) return miss(DrawRecipeMiss::ObjectsGone);
    const std::size_t expectedViews = state.colors.size() + (recipe.hasDepth ? 1u : 0u);
    if (recipe.hasDepth != state.depth.has_value() || recipe.targets.size() != state.colors.size() || recipe.targetViews.size() != expectedViews) return miss(DrawRecipeMiss::TargetGone);
    auto resources = recipe.templateRef.lock();
    if (resources == nullptr) return miss(DrawRecipeMiss::TemplateGone);
    if (!resources->Reusable() || resources->HeapStage()) return miss(DrawRecipeMiss::Moved);
    const bool spaceRoute = recipe.spaceReuse;
    if (spaceRoute != resources->SpaceReuse() || (spaceRoute && !RuntimeSwitches::Set(RuntimeSwitch::RecordHeavyDraws))) return miss(DrawRecipeMiss::TemplateGone);
    if (spaceRoute && resources->ChecksFaults() && std::ranges::any_of(shaders, [](const CompiledShader& shader) { return shader.program != nullptr && !shader.program->stencilReferences.empty(); })) return miss(DrawRecipeMiss::NotRecordable);
    std::optional<FullDrawSteps> steps;
    if (spaceRoute && recording && fullDrawCensus()) timer.steps = &steps.emplace();
    // A recipe with multisampled targets (APS5_MSAA_RECORDS): the draw must still have them where the
    // recipe has them, none staged, and must not sample one it renders to (the full draw refuses that).
    const bool multisampleRecipe = !recipe.multisampled.empty();
    if (multisampleRecipe) {
        if (!RuntimeSwitches::Set(RuntimeSwitch::MsaaRecords) || recipe.multisampled.size() != state.colors.size()) return miss(DrawRecipeMiss::TargetGone);
        const auto& sources = resources->MultisampleSources();
        for (std::size_t index = 0; index < state.colors.size(); ++index) {
            const auto& color = state.colors[index];
            if ((color.samples > 1) != (recipe.multisampled[index] != 0u)) return miss(DrawRecipeMiss::TargetGone);
            if (color.samples > 1 && (color.staged || std::count(sources.begin(), sources.end(), color.surfaceAddress) != 0)) return miss(DrawRecipeMiss::TargetGone);
        }
    }
    const bool lean = resources->RecordLean();
    DrawInputs inputs;
    if (recording) {
        const bool inputsKept = RuntimeSwitches::Set(RuntimeSwitch::XpInputsAsIs);
        if (!inputsKept && !inputsAsIs().empty()) inputsAsIs().clear();
        if (!inputsKept || !takeInputsAsIs(context, record, draw, inputs)) {
            const auto today = [&] {
                const auto plan = planDraw(context, state, draw, shaders, &recipe, outcome, timer);
                if (plan.nothing) {
                    inputs = {};
                    inputs.nothing = true;
                } else {
                    inputs = prepareDrawInputs(context, state, draw, shaders, plan, timer, timing, &recipe);
                }
            };
            bool ready = false;
            if (preparedInputs != nullptr && RuntimeSwitches::Set(RuntimeSwitch::DirectInputs)) {
                if (record.verifyInputs) {
                    today();
                    std::string difference;
                    record.effects.inputsCapacity = preparedInputs->exactCapacity;
                    record.effects.inputs = verifyPreparedInputs(context, state, draw, shaders, recipe, *preparedInputs, inputs, timer, timing, difference);
                    if (!difference.empty()) noteInputsDifference(difference);
                    ready = true;
                } else {
                    auto inputsCheck = committerTrustCheck(*resources, CommitterTrustFamily::Inputs, CommitterTrustSite::RecordInputs);
                    record.effects.inputs = takePreparedInputs(context, state, draw, shaders, recipe, *preparedInputs, inputs, timer, timing, false, false, &inputsCheck);
                    ready = record.effects.inputs == DrawRecordInputs::Used || record.effects.inputs == DrawRecordInputs::Rechecked;
                }
            }
            if (!ready) today();
            if (inputsKept) keepInputsAsIs(record, draw, inputs);
        }
        if (inputs.nothing) {
            result.recorded = true;
            return result;
        }
        for (const auto& color : state.colors) {
            if (color.fmaskAddress != 0) GuardOneFragmentFmask(context, color);
        }
    }
    timer.lap(FullDrawStep::Inputs);
    std::vector<std::shared_ptr<StorageTexture>> targets;
    targets.reserve(recipe.targets.size());
    // The owners and images of the multisampled targets, by color index (owners) and in order (images).
    std::vector<std::shared_ptr<void>> multisampleOwners(multisampleRecipe ? recipe.targets.size() : 0u);
    std::vector<VkImage> multisampleImages;
    const auto holdUnder = [&](const DrawRecordPass& pass) { return recordPassHold(pass, *recorder, recipe, state); };
    const auto* pass = &record.pass;
    auto hold = holdUnder(*pass);
    const SharedRecordPass* shared = nullptr;
    if (hold != DrawRecordHold::Held && RuntimeSwitches::Set(RuntimeSwitch::DirectPassTargets) && !multisampleRecipe) {
        const auto& kept = sharedRecordPass();
        if (sharedRecordPassServes(kept, state, recipe) && holdUnder(kept.pass) == DrawRecordHold::Held) shared = &kept;
    }
    if (shared != nullptr) pass = &shared->pass;
    bool held = hold == DrawRecordHold::Held || shared != nullptr;
    if (multisampleRecipe && !RuntimeSwitches::Set(RuntimeSwitch::MsaaRecordHold)) held = false;
    for (std::size_t index = 0; held && index < recipe.targets.size(); ++index) {
        if (multisampleRecipe && recipe.multisampled[index] != 0u) {
            VkImage image = VK_NULL_HANDLE;
            const auto view = MultisampleTargetView(context, state.colors[index], multisampleOwners[index], &image);
            if (view != recipe.targetViews[index] || multisampleOwners[index] == nullptr) {
                held = false;
                hold = DrawRecordHold::Target;
                break;
            }
            multisampleImages.push_back(image);
            targets.push_back(nullptr);
            continue;
        }
        auto stored = recipe.targets[index].lock();
        if (stored == nullptr || (shared != nullptr && stored.get() != shared->targets[index]) || stored->UploadVersion() != pass->versions[index] || !stored->Cached() || !StorageImageServesKeys(*stored, SurfaceForTarget(state.colors[index]).dccAddress)) {
            held = false;
            if (shared == nullptr) hold = DrawRecordHold::Target;
            shared = nullptr;
            break;
        }
        targets.push_back(std::move(stored));
    }
    record.effects.hold = hold;
    record.effects.sharedHold = held && shared != nullptr;
    if (!held) {
        targets.clear();
        multisampleImages.clear();
        for (auto& owner : multisampleOwners) owner.reset();
        if (failed(DrawRecipeMiss::TargetGone)) return miss(DrawRecipeMiss::TargetGone);
        for (std::size_t index = 0; index < recipe.targets.size(); ++index) {
            timer.phase(PhaseSetup);
            if (multisampleRecipe && recipe.multisampled[index] != 0u) {
                // As the full draw: the target's view is asked for per draw; the hit stands while it is the recipe's.
                VkImage image = VK_NULL_HANDLE;
                const auto view = MultisampleTargetView(context, state.colors[index], multisampleOwners[index], &image);
                timer.phase(PhaseReadTarget);
                if (view != recipe.targetViews[index] || multisampleOwners[index] == nullptr) return miss(DrawRecipeMiss::TargetGone);
                multisampleImages.push_back(image);
                targets.push_back(nullptr);
                continue;
            }
            auto stored = recipe.targets[index].lock();
            if (stored == nullptr || !StorageImageCached(context, stored.get()) || !StorageImageServesKeys(*stored, SurfaceForTarget(state.colors[index]).dccAddress)) return miss(DrawRecipeMiss::TargetGone);
            auto resident = refreshResidentTarget(context, state, state.colors[index], outcome, profile, [&] {
                if (state.colors[index].arrayLastSlice != 0) stored->RefreshLayers(state.colors[index].arraySlice, 1);
                else stored->Refresh();
                return stored;
            });
            timer.phase(PhaseReadTarget);
            if (resident == nullptr) return miss(DrawRecipeMiss::TargetGone);
            targets.push_back(std::move(resident));
        }
        if (recipe.hasDepth) {
            timer.phase(PhaseSetup);
            const auto depthView = DepthSurfaceViewWithStencilWriteIntent(context, *state.depth, state.depthWrite, state.stencilTest, writesStencilTarget(state), overwrittenStencil(state));
            timer.phase(PhaseReadTarget);
            if (recipe.targetViews.back() != depthView) return miss(DrawRecipeMiss::TargetGone);
        }
    }
    timer.phase(PhasePrepare);
    timer.lap(FullDrawStep::Targets);
    if (spaceRoute) {
        const auto taken = takeRecordSpace(*resources, space.snapshots);
        timer.lap(FullDrawStep::Space);
        if (taken != RecordSpaceOutcome::Served) {
            if (recording) NoteRecordHeavy(recordSpaceEvent(taken));
            timer.phase(PhaseLookup);
            return miss(DrawRecipeMiss::TemplateGone);
        }
        space.resources = resources;
        space.recorder = recorder;
    }
    bool proofTrusted = false;
    const auto proved = [&] {
        const bool injected = failed(DrawRecipeMiss::Proof);
        return proveRecordTemplate(record, *resources, shaders, result, injected, &proofTrusted);
    };
    if (!proved()) {
        noteFailedProof(context, ProofSite::Record, result.proof, recording ? &recipe.key : nullptr);
        if (recording) {
            SharedResourceCache().Remove(recipe.key, resources.get());
            countCache(&DrawProfile::cacheInvalidated);
            resources->KeepStorageImageBytes(*recorder);
            recorder->Keep(std::move(resources));
        }
        timer.phase(PhaseLookup);
        return miss(DrawRecipeMiss::Proof);
    }
    timer.lap(FullDrawStep::Proof);
    // Prepared buffers count only for the template whose plan they were prepared from. A verify
    // compares them with today's way, which then draws.
    const ShaderResources::PreparedBuffers* bound = nullptr;
    const ShaderResources::PreparedBuffers* checked = nullptr;
    if (recording && prepared != nullptr && RuntimeSwitches::Set(RuntimeSwitch::DirectBuffers)) {
        if (prepared->plan == nullptr || prepared->plan != resources->DirectBufferPlan(shaders)) record.effects.buffers = DrawRecordBuffers::PlanDiffers;
        else if (record.verifyBuffers) {
            checked = prepared;
            record.effects.bufferShortcuts = prepared->shortcuts;
        } else bound = prepared;
    }
    std::vector<ShaderResources::MovedBuffer> moved;
    {
        bool patched = false;
        auto ticket = record.ticket;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> probedRanges;
        if (recording) {
            ticket.ranges = &record.ticketRanges;
        } else {
            probedRanges = record.ticketRanges;
            ticket.ranges = &probedRanges;
        }
        ticket.checked = 0;
        ticket.trusted = 0;
        std::optional<std::vector<ShaderResources::MovedBuffer>> found;
        // A verify runs the prepared moved-range check first, on a copy of the ticket.
        auto preparedTicket = ticket;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> preparedRanges;
        bool preparedClear = false;
        bool preparedBehind = false;
        if (checked != nullptr) {
            preparedRanges = *ticket.ranges;
            preparedTicket.ranges = &preparedRanges;
            preparedClear = resources->PreparedMovesClear(*checked, *recorder, &preparedTicket) && resources->PreparedInPlaceHolds(*checked, *recorder, preparedBehind);
        }
        const bool trusted = bound != nullptr && RuntimeSwitches::Set(RuntimeSwitch::XpTrustCommitter) && (bound->forget & 1u) == 0 && GuestMemory::ForgetSerial() == bound->forget;
        if (trusted) NoteCommitterTrusted();
        auto buffersCheck = bound != nullptr ? committerTrustCheck(*resources, CommitterTrustFamily::Buffers, CommitterTrustSite::RecordBuffers) : CommitterTrustCheck{};
        if (bound != nullptr) {
            const bool boundMapped = buffersCheck.Armed() && (bound->forget & 1u) == 0 && GuestMemory::ForgetSerial() == bound->forget;
            const bool recordValidation = WriteValidation::Session() != 0;
            const auto movesClear = [&] {
                if (trusted || (boundMapped && buffersCheck.Skip())) return true;
                const auto byWindows = recordValidation ? resources->PreparedMovesClearByWindows(*bound, *recorder, &ticket, WriteValidation::Test::RecordMoves) : std::optional<bool>{};
                if (byWindows.has_value() ? *byWindows : resources->PreparedMovesClear(*bound, *recorder, &ticket)) return true;
                if (boundMapped) buffersCheck.Fail();
                return false;
            };
            if (movesClear() && resources->PreparedInPlaceHolds(*bound, *recorder, preparedBehind, &buffersCheck, recordValidation ? std::optional<WriteValidation::Test>{WriteValidation::Test::RecordInPlace} : std::optional<WriteValidation::Test>{})) {
                found.emplace(bound->moved.begin(), bound->moved.end());
                for (auto& entry : *found) {
                    if (entry.inPlace != VK_NULL_HANDLE) entry.behind = recorder->PendingWriteOverlaps(entry.begin, entry.bytes);
                }
            } else {
                record.effects.buffers = DrawRecordBuffers::MovedHazard;
                bound = nullptr;
                ticket.checked = 0;
                ticket.trusted = 0;
            }
        }
        if (!found.has_value()) found = resources->MovedReadOnlyBuffers(shaders, *recorder, &state.color, &patched, &ticket);
        if (checked != nullptr) {
            auto& difference = record.effects.bufferDifference;
            if (found.has_value()) {
                difference = ShaderResources::ComparePreparedMoves(*checked, *found);
                const bool inPlace = std::ranges::any_of(*found, [](const ShaderResources::MovedBuffer& entry) { return entry.inPlace != VK_NULL_HANDLE; });
                if (difference.what == nullptr && !inPlace && !checked->inPlace) difference = ShaderResources::ComparePreparedTickets(preparedClear ? &preparedTicket : nullptr, ticket);
            } else if (preparedClear && !checked->inPlace) {
                difference = {"moved buffers", "today's way refuses the draw's moved buffers, which the finisher prepared and the committer's tests find clear"};
            } else {
                record.effects.buffers = DrawRecordBuffers::Checked;
            }
            if (difference.what != nullptr) {
                record.effects.buffers = DrawRecordBuffers::Differs;
                checked = nullptr;
            }
        }
        if (!found.has_value() || failed(DrawRecipeMiss::Moved)) {
            if (recording) record.ticket.valid = false;
            timer.phase(PhaseLookup);
            return miss(DrawRecipeMiss::Moved);
        }
        if (recording) {
            ticket.ranges = nullptr;
            record.ticket = ticket;
            record.effects.movedChecked = ticket.checked;
            record.effects.movedTrusted = ticket.trusted;
        }
        moved = std::move(*found);
    }
    if (recording) {
        CheckBufferAliases(shaders, state.color, draw.indexAddress, inputs.indexBytes);
        if (recipe.hasDepth && writesDepthTarget(state)) NoteDepthSurfaceWrite(context, *state.depth);
        if (const auto present = Recorder::Presents(); record.touchedPresent != present && !RuntimeSwitches::Set(RuntimeSwitch::XpCommitFloor)) {
            if (!RuntimeSwitches::Set(RuntimeSwitch::TemplateTouch) || resources->TouchedPresent() != present) {
                SharedResourceCache().Touch(recipe.key);
                resources->NoteTouchedPresent(present);
            }
            record.touchedPresent = present;
        }
        countCache(&DrawProfile::cacheHits);
    }
    if (spaceRoute && recording && verifyAddressDrawReuse()) verifyServedSpace(context, state, draw, shaders, space.snapshots, inputs.indexBytes, *resources);
    space.settled = true;
    timer.lap(FullDrawStep::Moved);
    timer.phase(PhaseLookup);
    auto pipeline = recipe.pipeline.lock();
    if (pipeline == nullptr || failed(DrawRecipeMiss::ObjectsGone)) return miss(DrawRecipeMiss::ObjectsGone);
    std::vector<std::shared_ptr<void>> framebufferOwners(targets.begin(), targets.end());
    for (std::size_t index = 0; index < multisampleOwners.size(); ++index) {
        if (multisampleOwners[index] != nullptr) framebufferOwners[index] = multisampleOwners[index];
    }
    auto framebuffer = recipe.framebuffer.lock();
    if (framebuffer == nullptr) framebuffer = pipeline->AcquireFramebuffer(recipe.targetViews, framebufferOwners, state.renderExtent, state.renderLayers);
    const ShaderResources::PushPlan* push = nullptr;
    std::shared_ptr<Pipeline> pushPipeline;
    std::shared_ptr<Framebuffer> pushFramebuffer;
    const bool stencilReferences = std::ranges::any_of(shaders, [](const CompiledShader& shader) { return shader.program != nullptr && !shader.program->stencilReferences.empty(); });
    if (lean && !stencilReferences && !UsesMeshWindow(shaders) && !pipeline->DescriptorHeap()) push = resources->PushLayouts();
    bool movesPushed = false;
    if (push != nullptr) {
        if (record.pushTemplate == resources.get()) {
            pushPipeline = record.pushPipeline.lock();
            pushFramebuffer = record.pushFramebuffer.lock();
        }
        movesPushed = std::ranges::any_of(moved, [&](const ShaderResources::MovedBuffer& entry) { return entry.allocation < push->allocationSets.size() && push->allocationSets[entry.allocation] == push->set; });
        if (pushPipeline == nullptr && movesPushed) {
            const State& pipelineState = recipe.masked.has_value() ? *recipe.masked : state;
            pushPipeline = CachedPipeline(context, pipelineState, recipe.vertexInput, push->layouts, shaders, VK_IMAGE_LAYOUT_GENERAL, nullptr, nullptr, PipelineDefer::Skip);
            record.pushPipeline = pushPipeline;
            record.pushTemplate = resources.get();
            pushFramebuffer.reset();
        }
        if (pushPipeline != nullptr && pushFramebuffer == nullptr) {
            pushFramebuffer = pushPipeline->AcquireFramebuffer(recipe.targetViews, framebufferOwners, state.renderExtent, state.renderLayers);
            record.pushFramebuffer = pushFramebuffer;
        }
        if (pushPipeline == nullptr) push = nullptr;
    }
    std::shared_ptr<Pipeline> dynamicPipeline;
    std::shared_ptr<Framebuffer> dynamicFramebuffer;
    std::shared_ptr<Pipeline> pushDynamicPipeline;
    std::shared_ptr<Framebuffer> pushDynamicFramebuffer;
    using DynamicLeave = ShaderResources::DynamicLeave;
    const bool dynamicAsked = recording && RuntimeSwitches::Set(RuntimeSwitch::CommitBindings) && RuntimeSwitches::Set(RuntimeSwitch::CommitBindingsDynamic);
    const bool dynamicRoute = dynamicAsked && bound != nullptr;
    auto plainLeave = DynamicLeave::None;
    auto pushedLeave = DynamicLeave::None;
    if (dynamicRoute) {
        const auto acquire = [&](const PipelineSetLayouts& layouts, std::weak_ptr<Pipeline>& keptPipeline, std::weak_ptr<Framebuffer>& keptFramebuffer, const void*& keptTemplate, std::shared_ptr<Pipeline>& made, std::shared_ptr<Framebuffer>& madeFramebuffer) {
            if (keptTemplate == resources.get()) {
                made = keptPipeline.lock();
                madeFramebuffer = keptFramebuffer.lock();
            }
            if (made == nullptr) {
                const State& pipelineState = recipe.masked.has_value() ? *recipe.masked : state;
                made = CachedPipeline(context, pipelineState, recipe.vertexInput, layouts, shaders, VK_IMAGE_LAYOUT_GENERAL, nullptr, nullptr, PipelineDefer::Skip);
                keptPipeline = made;
                keptTemplate = resources.get();
                madeFramebuffer.reset();
            }
            if (made != nullptr && madeFramebuffer == nullptr) {
                madeFramebuffer = made->AcquireFramebuffer(recipe.targetViews, framebufferOwners, state.renderExtent, state.renderLayers);
                keptFramebuffer = madeFramebuffer;
            }
        };
        plainLeave = record.bindingsRefused ? DynamicLeave::Refused : stencilReferences || UsesMeshWindow(shaders) ? DynamicLeave::Stages : pipeline->DescriptorHeap() ? DynamicLeave::HeapPipeline : DynamicLeave::None;
        pushedLeave = plainLeave;
        if (plainLeave == DynamicLeave::None) {
            plainLeave = push != nullptr && movesPushed ? DynamicLeave::PushHeld : resources->DynamicLeaveFor(false);
            if (plainLeave == DynamicLeave::None) {
                acquire(resources->DynamicLayouts()->pipelineLayouts, record.dynamicPipeline, record.dynamicFramebuffer, record.dynamicTemplate, dynamicPipeline, dynamicFramebuffer);
                if (dynamicPipeline == nullptr || dynamicFramebuffer == nullptr) plainLeave = DynamicLeave::Pipeline;
            }
        }
        if (pushedLeave == DynamicLeave::None && push != nullptr && movesPushed) {
            const auto seen = record.repeatSeen;
            const bool repeatsFirst = RuntimeSwitches::Set(RuntimeSwitch::DirectRepeat) && seen != DrawRepeatClass::Template && seen != DrawRepeatClass::Stages && seen != DrawRepeatClass::Pipeline && seen != DrawRepeatClass::Targets && seen != DrawRepeatClass::Buffers && seen != DrawRepeatClass::Inputs;
            pushedLeave = resources->DynamicLeaveFor(true);
            if (pushedLeave == DynamicLeave::None && repeatsFirst) pushedLeave = DynamicLeave::Repeats;
            if (pushedLeave == DynamicLeave::None && !record.secondSetWritten) pushedLeave = DynamicLeave::First;
            if (pushedLeave == DynamicLeave::None) {
                acquire(*resources->DynamicPushLayouts(), record.pushDynamicPipeline, record.pushDynamicFramebuffer, record.pushDynamicTemplate, pushDynamicPipeline, pushDynamicFramebuffer);
                if (pushDynamicPipeline == nullptr || pushDynamicFramebuffer == nullptr) pushedLeave = DynamicLeave::Pipeline;
            }
        }
    }
    record.effects.unpushed = lean && push == nullptr && !moved.empty();
    DrawFragment* usableFragment = nullptr;
    if (fragment != nullptr && recording) {
        auto& note = record.effects.fragment;
        note.offered = true;
        const Pipeline* expected = fragment->pushing ? pushPipeline.get() : pipeline.get();
        auto drop = DrawFragmentDrop::None;
        if (record.fragmentsRefused) drop = DrawFragmentDrop::Refused;
        else if (fragment->device != context.device || fragment->resources.get() != resources.get() || fragment->push != push || fragment->pipeline.get() != expected || fragment->bindings == nullptr) drop = DrawFragmentDrop::Objects;
        else if (bound == nullptr) drop = DrawFragmentDrop::Selection;
        else if (record.effects.inputs != DrawRecordInputs::Used && record.effects.inputs != DrawRecordInputs::Rechecked) drop = DrawFragmentDrop::Inputs;
        else if (!lean || dynamicRoute || trace != nullptr || stencilReferences || UsesMeshWindow(shaders) || !fragmentModes()) drop = DrawFragmentDrop::Modes;
        note.drop = drop;
        if (drop == DrawFragmentDrop::None) usableFragment = fragment;
    }
    timer.phase(PhasePipeline);
    timer.lap(FullDrawStep::Pipeline);
    if (!recording) {
        answer->filled = true;
        answer->resources = resources.get();
        answer->pipeline = pipeline.get();
        answer->framebuffer = framebuffer.get();
        answer->key = recipe.key;
        answer->views = recipe.targetViews;
        answer->targets.clear();
        for (const auto& target : targets) answer->targets.push_back(target.get());
        answer->passKey = recipe.passKey;
        answer->moved.clear();
        answer->movedSizes.clear();
        answer->movedWords.clear();
        for (const auto& entry : moved) {
            answer->moved.emplace_back(entry.allocation, entry.address);
            answer->movedSizes.push_back(entry.size);
            answer->movedWords.emplace_back(entry.words.begin(), entry.words.end());
        }
        answer->pushStages = recipe.pushStages;
        answer->push = {};
        if (recipe.pushStages != 0) {
            answer->push = AssemblePushConstants(shaders);
            resources->PatchPushConstants(answer->push);
        }
        answer->vertexInput = recipe.vertexInput;
        answer->fragmentOutputs = recipe.fragmentOutputs;
        answer->shaderStages = recipe.shaderStages;
        answer->hasDepth = recipe.hasDepth;
        answer->pushed = false;
        answer->bound = false;
        answer->bindings.clear();
        if (lean) {
            std::vector<std::size_t> selected;
            const auto bindings = push != nullptr ? resources->PreparePushedDrawBindings(*recorder, moved, *push, &selected) : resources->PrepareDrawBindings(*recorder, moved, false, nullptr, false, &selected);
            resources->DescribeDrawBindings(bindings.get(), selected, push, answer->bindings);
            answer->bound = true;
            answer->pushed = bindings != nullptr && bindings->pushedSet >= 0;
        }
        answer->pushPipeline = answer->pushed ? pushPipeline.get() : nullptr;
        answer->pushFramebuffer = answer->pushed ? pushFramebuffer.get() : nullptr;
        answer->access.reset();
        if (!resources->HoldsLease() && !resources->UsesAddressTable() && !PassAccessMemoVerify() && passAccessMatches(record.access, recorder, *resources, targets)) answer->access = record.access;
        const Pipeline* boundPipeline = answer->pushed ? pushPipeline.get() : pipeline.get();
        const Framebuffer* boundFramebuffer = answer->pushed ? pushFramebuffer.get() : framebuffer.get();
        answer->keptOnce = lean && keepsOnce(&record.keep, *recorder, resources.get(), boundPipeline, boundFramebuffer, targets, !resources->ChecksFaults());
        answer->keep = record.keep;
        answer->passNotes = lean;
        record.effects.passHeld = held;
        return result;
    }
    IndirectRecord indirectRecord;
    if (draw.indirect && (inputs.listRestartPrimitive != 0 || inputs.expandedIndexCount != 0 || !inputs.restartTable.empty() || !gpuIndirectRecord(context, *recorder, *draw.indirect, indirectRecord))) return miss(DrawRecipeMiss::NotRecordable);
    outcome.completion = spaceRoute ? resources->NeedsCompletion() : resources->ChecksFaults();
    outcome.recorded = true;
    std::vector<StorageTexture*> residents;
    residents.reserve(targets.size());
    for (const auto& target : targets) residents.push_back(target.get());
    const void* plainPipeline = pipeline.get();
    const void* plainFramebuffer = framebuffer.get();
    const bool inputsPlain = !inputs.nothing && inputs.indices == nullptr && inputs.vertexBuffers.empty() && inputs.restartTable.empty();
    RecordedDraw drawn;
    drawn.recorder = recorder;
    drawn.completion = outcome.completion;
    drawn.spaceKeep = spaceRoute ? &space.keep : nullptr;
    drawn.resources = std::move(resources);
    drawn.pipeline = std::move(pipeline);
    drawn.framebuffer = std::move(framebuffer);
    drawn.targetViews = recipe.targetViews;
    drawn.targets = std::move(targets);
    for (const auto& color : state.colors) drawn.targetSlices.emplace_back(color.arraySlice, color.arrayLastSlice != 0 ? color.layers : 1u);
    drawn.moved = moved;
    drawn.indirect = draw.indirect ? &indirectRecord : nullptr;
    drawn.pushBytes = nullptr;
    drawn.pushStages = recipe.pushStages;
    drawn.depthLocations = passDepthLocations(context, state, *drawn.pipeline);
    drawn.access = &record.access;
    if (lean) drawn.keeping = {&record.keep, true, &record.effects.notesSkipped, &record.effects.keptOnce};
    drawn.push = push;
    drawn.pushPipeline = std::move(pushPipeline);
    drawn.pushFramebuffer = std::move(pushFramebuffer);
    drawn.pushed = &record.effects.pushed;
    drawn.prepared = bound;
    drawn.preparedCheck = checked;
    drawn.buffers = &record.effects.buffers;
    drawn.bufferDifference = &record.effects.bufferDifference;
    ShaderResources::BindingRoute route;
    if (bound != nullptr) {
        auto& noted = record.effects.bindings;
        drawn.bindings = &noted;
        noted.timed = CommitBindingTimings() && (++commitBindingTimingOrdinal & 7u) == 0;
        if (RuntimeSwitches::Set(RuntimeSwitch::CommitBindings) && !record.bindingsRefused) {
            route.placed = bound->placed != nullptr;
            route.stamps = RuntimeSwitches::Set(RuntimeSwitch::CommitBindingsStamps);
            route.sets = RuntimeSwitches::Set(RuntimeSwitch::CommitBindingsSets);
            route.dynamic = dynamicPipeline != nullptr && dynamicFramebuffer != nullptr;
            route.pushDynamic = pushDynamicPipeline != nullptr && pushDynamicFramebuffer != nullptr;
            route.plainLeave = plainLeave;
            route.pushedLeave = pushedLeave;
            const auto every = CommitBindingsVerifyEvery();
            noted.verify = every != 0 && (route.placed || route.stamps || route.sets || route.dynamic || route.pushDynamic) && ++commitBindingVerifyOrdinal % every == 0;
            route.verifyStamps = noted.verify;
            if (route.placed || route.stamps || route.sets || dynamicRoute) drawn.route = &route;
            if (route.dynamic) {
                drawn.dynamicPipeline = dynamicPipeline;
                drawn.dynamicFramebuffer = dynamicFramebuffer;
            }
            if (route.pushDynamic) {
                drawn.pushDynamicPipeline = pushDynamicPipeline;
                drawn.pushDynamicFramebuffer = pushDynamicFramebuffer;
            }
            noted.routed = drawn.route != nullptr;
        }
        if (dynamicRoute && record.bindingsRefused) noted.dynamicLeave = static_cast<std::uint8_t>(DynamicLeave::Refused);
    } else if (dynamicAsked) {
        drawn.bindings = &record.effects.bindings;
        record.effects.bindings.dynamicLeave = static_cast<std::uint8_t>(DynamicLeave::Prepared);
    }
    std::vector<std::shared_ptr<void>> expressMultisample;
    std::vector<VkImage> expressMultisampleImages;
    if (multisampleRecipe && RuntimeSwitches::Set(RuntimeSwitch::ExpressCommits)) {
        expressMultisample = multisampleOwners;
        expressMultisampleImages.assign(recipe.targets.size(), VK_NULL_HANDLE);
        std::size_t next = 0;
        for (std::size_t index = 0; index < recipe.targets.size() && index < recipe.multisampled.size(); ++index) {
            if (recipe.multisampled[index] != 0u && next < multisampleImages.size()) expressMultisampleImages[index] = multisampleImages[next++];
        }
    }
    drawn.multisampleImages = std::move(multisampleImages);
    for (auto& owner : multisampleOwners) {
        if (owner != nullptr) drawn.multisampleTargets.push_back(std::move(owner));
    }
    if (RuntimeSwitches::Set(RuntimeSwitch::XpBindingsAsIs)) drawn.bindingsAsIs = &bindingsAsIs()[&record];
    else if (!bindingsAsIs().empty()) bindingsAsIs().clear();
    if (drawn.route != nullptr) {
        if (const auto hook = commitBindingsHook.exchange(nullptr, std::memory_order_acq_rel); hook != nullptr) hook(record);
    }
    drawn.fragment = usableFragment;
    drawn.fragmentNote = &record.effects.fragment;
    if (usableFragment != nullptr) {
        const auto every = CommitFragmentsVerifyEvery();
        drawn.fragmentVerify = every != 0 && ++commitFragmentVerifyOrdinal % every == 0;
        if (std::exchange(record.feedVerify, false)) drawn.fragmentVerify = true;
    }
    const bool expressWanted = usableFragment != nullptr && RuntimeSwitches::Set(RuntimeSwitch::ExpressCommits) && lean && !record.expressRefused && PixelFunctionLdsSubgroup(shaders) == 0u;
    if (!expressWanted && usableFragment != nullptr && RuntimeSwitches::Set(RuntimeSwitch::ExpressCommits)) {
        record.express.reset();
        auto& noted = record.effects.fragment.expressCapture;
        if (!lean) noted = DrawExpressCapture::NotLean;
        else if (record.expressRefused) noted = DrawExpressCapture::Refused;
        else noted = DrawExpressCapture::FunctionLds;
    }
    std::shared_ptr<Framebuffer> expressFramebuffer;
    std::vector<std::shared_ptr<StorageTexture>> expressTargets;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> expressSlices;
    if (expressWanted && !drawn.resources->ExpressReady() && !(RuntimeSwitches::Set(RuntimeSwitch::ExpressKeepState) && drawn.resources->ExpressWatchable())) {
        record.effects.fragment.expressCapture = DrawExpressCapture::NotProved;
        record.express.reset();
    } else if (expressWanted) {
        expressFramebuffer = usableFragment->pushing ? drawn.pushFramebuffer : drawn.framebuffer;
        expressTargets = drawn.targets;
        expressSlices = drawn.targetSlices;
    }
    const std::weak_ptr<Pipeline> blockPipeline = drawn.pipeline;
    const std::weak_ptr<Pipeline> blockPushPipeline = drawn.pushPipeline;
    const void* blockPipelineKey = drawn.pipeline.get();
    const void* blockPushKey = drawn.pushPipeline.get();
    recordDraw(context, state, draw, shaders, inputs, drawn, outcome, timer, ownWaitedMs, trace);
    if (RuntimeSwitches::Set(RuntimeSwitch::CommitFragments) || record.fragmentResources != nullptr) publishFragmentBlock(context, record, drawn.resources, blockPipeline, blockPushPipeline, blockPipelineKey, blockPushKey, push, lean && bound != nullptr && !stencilReferences && !UsesMeshWindow(shaders), lean && bound == nullptr && !stencilReferences && !UsesMeshWindow(shaders));
    if (RuntimeSwitches::Set(RuntimeSwitch::CommitFragments) && drawn.resources != nullptr) {
        auto& blockMiss = record.effects.fragment.blockMiss;
        if (!lean) blockMiss = static_cast<DrawFragmentBlockMiss>(std::max<std::uint8_t>(drawn.resources->LeanMiss(), 1u));
        else if (bound == nullptr) blockMiss = DrawFragmentBlockMiss::NoBuffers;
        else if (stencilReferences) blockMiss = DrawFragmentBlockMiss::Stencil;
        else if (UsesMeshWindow(shaders)) blockMiss = DrawFragmentBlockMiss::MeshWindow;
        else if (!drawn.resources->FragmentEligible()) blockMiss = DrawFragmentBlockMiss::Ineligible;
        else if (record.fragmentsRefused) blockMiss = DrawFragmentBlockMiss::Refused;
        else if (!fragmentModes()) blockMiss = DrawFragmentBlockMiss::Modes;
    }
    if (expressWanted && expressFramebuffer != nullptr) captureExpress(context, state, recipe, record, *recorder, drawn.resources, *usableFragment, std::move(expressFramebuffer), std::move(expressTargets), std::move(expressSlices), std::move(expressMultisample), std::move(expressMultisampleImages), proofTrusted);
    if (dynamicRoute && drawn.bindings != nullptr && drawn.bindings->pushed && drawn.bindings->written) record.secondSetWritten = true;
    record.effects.passHeld = held;
    if ((!multisampleRecipe || RuntimeSwitches::Set(RuntimeSwitch::MsaaRecordHold)) && recorder->RenderPassOpen() && GuestMemory::CollectEpoch() != 0) {
        auto& remembered = record.pass;
        remembered.valid = true;
        remembered.recorder = recorder->Id();
        remembered.generation = recorder->RenderPassGeneration();
        remembered.epoch = GuestMemory::CollectEpoch();
        remembered.proof = GuestMemory::ProofSerial();
        remembered.pending = StorageTexture::PendingSerial();
        remembered.depthSerial = DepthSurfacesSerial();
        remembered.depthReleased = DepthSurfacesReleased();
        remembered.versions.clear();
        for (const auto* target : residents) remembered.versions.push_back(target != nullptr ? target->UploadVersion() : 0u);
        if (!record.effects.sharedHold && RuntimeSwitches::Set(RuntimeSwitch::DirectPassTargets) && !multisampleRecipe) {
            auto& kept = sharedRecordPass();
            kept.pass = remembered;
            kept.colors.assign(state.colors.begin(), state.colors.end());
            kept.depth = state.depth;
            kept.depthWrite = state.depthWrite;
            kept.stencilTest = state.stencilTest;
            kept.stencilWritten = writesStencilTarget(state);
            kept.depthCopySource = state.depthCopySource;
            kept.views = record.recipe->targetViews;
            kept.targets.assign(residents.begin(), residents.end());
        }
    } else {
        record.pass.valid = false;
    }
    if (repeatBlocks) fillRepeatBlock(context, state, shaders, record, recipe, *drawn.resources, plainPipeline, plainFramebuffer, drawn.pushPipeline.get(), drawn.pushFramebuffer.get(), residents, push, bound != nullptr && bound->inPlace ? nullptr : bound, inputsPlain, notesBefore, multisampleRecipe, *recorder);
    if (dynamicRoute && repeatBlocks && record.repeat != nullptr) record.repeatSeen = record.repeat->simple;
    if (bound != nullptr) ShaderResources::NotePreparedInPlaceTaken(*bound);
    verifyMovedInPlace(context, state, draw, shaders, inputs.indexBytes, *drawn.resources, moved);
    timing.Mark("draw_and_resource_release");
    reportDrawEnd(state, timer, nullptr, outcome, waitedBefore, ownWaitedMs, " recorded from record");
    if (spaceRoute) {
        countAddressDrawReuse(AddressDrawReuse::Reused);
        if (steps.has_value()) {
            steps->Lap(FullDrawStep::Finish);
            noteRecordHeavyServed(*steps);
        }
    }
    result.recorded = true;
    return result;
}

}

DrawRecipeOutcome DrawWithRecord(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, DrawRecordCommit& record, const DrawTraceContext* trace, DrawCommitView* answer, const ShaderResources::PreparedBuffers* prepared, PreparedInputs* preparedInputs, DrawFragment* fragment, std::span<const GuestMemorySnapshot> snapshots) {
    RecordSpaceHold space;
    space.snapshots = snapshots;
    try {
        return drawWithRecord(context, state, draw, shaders, record, trace, answer, prepared, preparedInputs, fragment, space);
    } catch (...) {
        space.KeepOnThrow();
        throw;
    }
}

namespace {

RepeatRefusal repeatHoldRefusal(DrawRecordHold hold) {
    switch (hold) {
    case DrawRecordHold::Epoch: return RepeatRefusal::Epoch;
    case DrawRecordHold::Proof: return RepeatRefusal::ProofSerial;
    case DrawRecordHold::Pending: return RepeatRefusal::Pending;
    case DrawRecordHold::Depth: return RepeatRefusal::Depth;
    case DrawRecordHold::Target: return RepeatRefusal::Target;
    default: return RepeatRefusal::Pass;
    }
}

bool repeatTargetsHeld(const DrawRepeatBlock& block, const DrawRecordPass& pass) {
    for (std::size_t index = 0; index < block.targets.size(); ++index) {
        const auto* target = block.targets[index];
        if (target->UploadVersion() != pass.versions[index] || !target->Cached() || !StorageImageServesKeys(*target, block.dccAddresses[index])) return false;
    }
    return true;
}

bool repeatStoresClear(const Recorder& recorder, const ShaderResources& resources, bool& queued) {
    queued = recorder.HasQueuedKeyStores() || recorder.HasQueuedStores();
    if (!queued) return true;
    if (!RuntimeSwitches::Set(RuntimeSwitch::DirectRepeatStores)) return false;
    const auto touches = [&](std::uint64_t begin, std::uint64_t end) {
        const auto bytes = static_cast<std::size_t>(end - begin);
        return resources.WritesOverlap(begin, bytes) || resources.ReadsOverlap(begin, bytes);
    };
    if (recorder.HasQueuedKeyStores() && recorder.AnyQueuedKeyStore(touches)) return false;
    return !recorder.HasQueuedStores() || !recorder.AnyQueuedStore(touches);
}

struct RepeatReadsStorage {};
struct RepeatAttachmentsStorage {};

std::uint64_t repeatOrdinal = 0;
std::atomic<RepeatHookForTest> repeatHook{nullptr};

}

void SetRepeatHookForTest(RepeatHookForTest hook) {
    repeatHook.store(hook, std::memory_order_release);
}

void SetCommitBindingsHookForTest(CommitBindingsHookForTest hook) {
    commitBindingsHook.store(hook, std::memory_order_release);
}

void SetExpressHookForTest(ExpressHookForTest hook) {
    expressHook.store(hook, std::memory_order_release);
}

std::uint64_t CommitBindingsVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_COMMIT_BINDINGS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

RepeatOutcome RepeatCommit(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, DrawRecordCommit& record, const ShaderResources::PreparedBuffers* prepared, PreparedInputs* preparedInputs, const DrawTraceContext* trace, DrawCommitView* describe, std::uint64_t verifyEvery) {
    RepeatOutcome out;
    const auto refuse = [&](RepeatRefusal why) {
        out.refusal = why;
        return out;
    };
    auto* block = record.repeat.get();
    if (block == nullptr) return refuse(RepeatRefusal::NoBlock);
    if (block->simple != DrawRepeatClass::Simple) {
        out.simple = block->simple;
        out.detail = block->detail;
        return refuse(RepeatRefusal::NotSimple);
    }
    auto* recorder = Recorder::Active();
    const auto* recipe = record.recipe.get();
    if (recipe == nullptr || block->recipe != recipe || block->state != &state || block->switches != RuntimeSwitches::Generation() || recorder == nullptr || context.recorder != recorder || context.device != block->device || recipe->device != context.device || context.descriptorCache != block->descriptorCache || draw.indirect || shaders.size() != block->stages || !RecordDraws() || DumpTargetLimit() != 0 || recordChecksQueued() || repeatExcludedModes()) return refuse(RepeatRefusal::Shape);
    for (std::size_t index = 0; index < shaders.size(); ++index) {
        if (shaders[index].program != block->programs[index]) return refuse(RepeatRefusal::Shape);
    }
    if (!keepsOnceFor<StorageTexture*>(&record.keep, *recorder, block->resources, block->pipeline, block->framebuffer, block->targets, true)) return refuse(RepeatRefusal::Batch);
    auto& resources = *block->resources;
    if (!resources.Reusable() || resources.HeapStage() || !resources.RecordLean()) return refuse(RepeatRefusal::Objects);
    const auto ownHold = recordPassHold(record.pass, *recorder, *recipe, state);
    const auto heldPass = [&]() -> const DrawRecordPass* {
        if (recordPassHold(record.pass, *recorder, *recipe, state) == DrawRecordHold::Held) return &record.pass;
        if (!RuntimeSwitches::Set(RuntimeSwitch::DirectRepeatShared) || !RuntimeSwitches::Set(RuntimeSwitch::DirectPassTargets)) return nullptr;
        const auto& kept = sharedRecordPass();
        if (!sharedRecordPassServes(kept, state, *recipe) || recordPassHold(kept.pass, *recorder, *recipe, state) != DrawRecordHold::Held || kept.targets.size() != block->targets.size()) return nullptr;
        for (std::size_t index = 0; index < block->targets.size(); ++index) {
            if (kept.targets[index] != block->targets[index]) return nullptr;
        }
        return &kept.pass;
    };
    const auto* holding = heldPass();
    if (holding == nullptr) return refuse(repeatHoldRefusal(ownHold));
    out.shared = holding != &record.pass;
    if (!repeatTargetsHeld(*block, *holding)) return refuse(RepeatRefusal::Target);
    const auto accessHeld = [&] {
        if (out.shared) return record.access.recorder == recorder && record.access.resources == &resources && std::ranges::equal(record.access.targets, block->targets, [](const void* kept, const StorageTexture* target) { return kept == target; });
        return passAccessMatchesFor<StorageTexture*>(record.access, recorder, resources, block->targets);
    };
    if (!accessHeld() || record.access.readsTarget) return refuse(RepeatRefusal::Access);
    if (!repeatStoresClear(*recorder, resources, out.stores)) return refuse(RepeatRefusal::Stores);
    if (StorageTexture::DrawnLinearPending()) return refuse(RepeatRefusal::Linear);
    out.sets = block->sets;
    if (prepared == nullptr || prepared->inPlace || !RuntimeSwitches::Set(RuntimeSwitch::DirectBuffers) || record.verifyBuffers || prepared->plan == nullptr || prepared->plan != resources.DirectBufferPlan(shaders)) return refuse(RepeatRefusal::Buffers);
    auto& push = block->bindings;
    if (prepared->blob.empty() || prepared->selected.size() != push.allocations.size()) return refuse(RepeatRefusal::Selection);
    for (std::size_t index = 0; index < push.allocations.size(); ++index) {
        if (prepared->selected[index].allocation != push.allocations[index] || prepared->selected[index].moved != block->moved[index]) return refuse(RepeatRefusal::Selection);
    }
    if (resources.RepeatGuards(*prepared) != ShaderResources::PreparedBuffers::Selection::Taken) return refuse(RepeatRefusal::DepthGuards);
    if (preparedInputs == nullptr || !RuntimeSwitches::Set(RuntimeSwitch::DirectInputs) || record.verifyInputs) return refuse(RepeatRefusal::Inputs);
    const bool memoRoute = !record.cleanProofs && !RuntimeSwitches::Set(RuntimeSwitch::CleanCurrency);
    if (describe == nullptr && repeatHook.load(std::memory_order_relaxed) != nullptr) {
        if (const auto hook = repeatHook.exchange(nullptr, std::memory_order_acq_rel); hook != nullptr) {
            hook(record);
            return RepeatCommit(context, state, draw, shaders, record, prepared, preparedInputs, trace, nullptr, verifyEvery);
        }
    }
    if (describe != nullptr) {
        auto& view = *describe;
        view.filled = true;
        view.resources = &resources;
        view.pipeline = block->recipePipeline;
        view.framebuffer = block->recipeFramebuffer;
        view.key = recipe->key;
        view.views = recipe->targetViews;
        view.targets.assign(block->targets.begin(), block->targets.end());
        view.passKey = recipe->passKey;
        view.moved.clear();
        view.movedSizes.clear();
        view.movedWords.clear();
        for (const auto& entry : prepared->moved) {
            view.moved.emplace_back(entry.allocation, entry.address);
            view.movedSizes.push_back(entry.size);
            view.movedWords.emplace_back(entry.words.begin(), entry.words.end());
        }
        view.pushStages = recipe->pushStages;
        view.push = {};
        if (recipe->pushStages != 0) {
            view.push = AssemblePushConstants(shaders);
            resources.PatchPushConstants(view.push);
        }
        view.vertexInput = recipe->vertexInput;
        view.fragmentOutputs = recipe->fragmentOutputs;
        view.shaderStages = recipe->shaderStages;
        view.hasDepth = recipe->hasDepth;
        auto slice = recorder->DrawUpload(prepared->blob.size());
        std::memcpy(slice.bytes, prepared->blob.data(), prepared->blob.size());
        ShaderResources::DrawBindings described;
        described.pushedSet = static_cast<std::int32_t>(block->push->set);
        described.pushSelected = push.allocations;
        for (const auto& item : prepared->selected) described.snapshots.push_back({item.begin, slice.buffer, slice.offset + item.offset, item.bytes});
        resources.DescribeDrawBindings(&described, push.allocations, block->push, view.bindings);
        out.pushDifference = resources.CompareRepeatPush(push, described, *block->push);
        view.bound = true;
        view.pushed = true;
        view.pushPipeline = block->pipeline;
        view.pushFramebuffer = block->framebuffer;
        view.access = record.access;
        view.keptOnce = true;
        view.keep = record.keep;
        view.passNotes = true;
        out.proofSkipped = memoRoute && resources.ProofHeld();
        out.unmovedSkipped = block->notesHeld && recorder->WriteNoteCount() == block->writeNotes;
        if (out.unmovedSkipped && !resources.RepeatUnmovedClear(*recorder, *prepared)) out.refuted = "unmoved ranges";
        out.taken = true;
        return out;
    }
    if (verifyEvery != 0 && ++repeatOrdinal % verifyEvery == 0) return refuse(RepeatRefusal::VerifyDue);
    const DepthCopySourceScope depthCopySource(state.depthCopySource);
    PerformanceTimer timing("Graphics.RepeatCommit");
    DrawTimer timer(false);
    recorder->BoundKeptBytes();
    if (record.keep.batch != recorder->OpenSerial() || !recorder->RenderPassOpen()) return refuse(RepeatRefusal::Batch);
    record.effects = {};
    DrawInputs inputs;
    auto inputsCheck = committerTrustCheck(resources, CommitterTrustFamily::Inputs, CommitterTrustSite::RecordInputs);
    out.inputs = takePreparedInputs(context, state, draw, shaders, *recipe, *preparedInputs, inputs, timer, timing, false, false, &inputsCheck);
    if (out.inputs != DrawRecordInputs::Used && out.inputs != DrawRecordInputs::Rechecked) return refuse(RepeatRefusal::Inputs);
    const auto giveBack = [&](RepeatRefusal why) {
        preparedInputs->vertexInput = std::move(inputs.vertexInput);
        preparedInputs->vertexHandles = std::move(inputs.vertexHandles);
        preparedInputs->vertexOffsets = std::move(inputs.vertexOffsets);
        preparedInputs->inPlaceReads = std::move(inputs.inPlaceReads);
        out.inputs = DrawRecordInputs::None;
        out.late = true;
        return refuse(why);
    };
    bool storesQueued = false;
    const auto* stillHolding = heldPass();
    if (record.keep.batch != recorder->OpenSerial() || stillHolding == nullptr || (stillHolding != &record.pass) != out.shared || !repeatTargetsHeld(*block, *stillHolding) || !accessHeld() || !repeatStoresClear(*recorder, resources, storesQueued) || StorageTexture::DrawnLinearPending()) return giveBack(RepeatRefusal::Changed);
    record.effects.inputs = out.inputs;
    out.proofSkipped = memoRoute && resources.ProofHeld();
    if (out.proofSkipped) {
        record.effects.proved = true;
    } else if (!proveRecordTemplate(record, resources, shaders, out.result, false)) {
        auto owner = recipe->templateRef.lock();
        SharedResourceCache().Remove(recipe->key, &resources);
        noteFailedProof(context, ProofSite::Repeat, out.result.proof, &recipe->key);
        countCache(&DrawProfile::cacheInvalidated);
        resources.KeepStorageImageBytes(*recorder);
        if (owner != nullptr) recorder->Keep(std::move(owner));
        out.taken = true;
        out.result.miss = DrawRecipeMiss::Proof;
        return out;
    }
    {
        auto ticket = record.ticket;
        ticket.ranges = &record.ticketRanges;
        ticket.checked = 0;
        ticket.trusted = 0;
        const auto movesByWindows = WriteValidation::Session() != 0 ? resources.PreparedMovesClearByWindows(*prepared, *recorder, &ticket, WriteValidation::Test::RecordMoves) : std::optional<bool>{};
        if (!(movesByWindows.has_value() ? *movesByWindows : resources.PreparedMovesClear(*prepared, *recorder, &ticket))) return giveBack(RepeatRefusal::MovedHazard);
        ticket.ranges = nullptr;
        record.ticket = ticket;
        record.effects.movedChecked = ticket.checked;
        record.effects.movedTrusted = ticket.trusted;
    }
    CheckBufferAliases(shaders, state.color, draw.indexAddress, inputs.indexBytes);
    if (const auto present = Recorder::Presents(); record.touchedPresent != present) {
        if (!RuntimeSwitches::Set(RuntimeSwitch::TemplateTouch) || resources.TouchedPresent() != present) {
            SharedResourceCache().Touch(recipe->key);
            resources.NoteTouchedPresent(present);
        }
        record.touchedPresent = present;
    }
    if (!repeatStoresClear(*recorder, resources, storesQueued)) return giveBack(RepeatRefusal::Stores);
    out.stores = out.stores || storesQueued;
    resources.RepeatRequire(prepared->moved);
    out.unmovedSkipped = block->notesHeld && recorder->WriteNoteCount() == block->writeNotes;
    if (!out.unmovedSkipped && !resources.RepeatUnmovedClear(*recorder, *prepared)) return giveBack(RepeatRefusal::UnmovedHazard);
    VkDeviceSize uploadOffset = 0;
    auto upload = resources.RepeatUpload(*recorder, *prepared, uploadOffset);
    if (block->upload != upload.get() || block->uploadBatch != recorder->OpenSerial() || block->uploadRecorder != recorder->Id()) {
        block->upload = upload.get();
        block->uploadBatch = recorder->OpenSerial();
        block->uploadRecorder = recorder->Id();
        recorder->Keep(upload);
    }
    if (RuntimeSwitches::Set(RuntimeSwitch::KeepBytesOnce)) recorder->KeepSnapshotBytes(upload.get(), upload->Bytes().size());
    else recorder->KeepBytes(upload.get(), upload->Bytes().size());
    record.effects.buffers = DrawRecordBuffers::Used;
    record.effects.pushed = true;
    const auto& memo = record.access;
    auto& passReads = HostThreadLocal<std::vector<std::pair<std::uint64_t, std::uint64_t>>, RepeatReadsStorage>();
    passReads.assign(memo.reads.begin(), memo.reads.end());
    passReads.insert(passReads.end(), inputs.inPlaceReads.begin(), inputs.inPlaceReads.end());
    auto& passAttachments = HostThreadLocal<std::vector<std::pair<VkImage, bool>>, RepeatAttachmentsStorage>();
    passAttachments.clear();
    for (const auto* target : block->targets) passAttachments.emplace_back(target->Image(), true);
    const PassAccess passAccess{passReads, memo.writes, memo.images, passAttachments, false, false, memo.writeOnly};
    resources.RequireGuestBuffers();
    const auto start = recorder->StartDrawPass(block->passKey, samplesAttachedDepth(*recorder, resources), block->fenced, passAccess, block->framebuffer != nullptr ? passRestartDepth(context, *block->framebuffer, state, resources) : VK_NULL_HANDLE);
    record.access.generation = recorder->RenderPassGeneration();
    const auto commands = start.commands;
    auto& recording = recorder->GraphicsState();
    const State* masked = recipe->masked.has_value() ? &*recipe->masked : nullptr;
    out.passBegun = !start.continued;
    out.passBreak = start.reason;
    auto drawTiming = Recorder::NoTiming;
    if (start.continued) {
        block->pipeline->Continue(recording, state, masked);
    } else {
        recorder->PrepareSampleSlot();
        PerformanceTrace::Count(PerformanceTrace::RenderPassScope);
        drawTiming = recorder->BeginGpuTiming(Recorder::CommandClass::Draw);
        if (Recorder::BarrierValidate()) {
            auto reads = resources.InPlaceReads();
            reads.insert(reads.end(), inputs.inPlaceReads.begin(), inputs.inPlaceReads.end());
            auto images = resources.StorageImages();
            for (const auto* target : block->targets) images.emplace_back(target->Image(), true);
            recorder->NoteAccess(Recorder::CommandClass::Draw, Recorder::Access{reads, resources.GpuWrites(), images, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | inputs.shaderStages, false});
        }
        if (start.barrier) {
            const bool passIndirect = RuntimeSwitches::Set(RuntimeSwitch::IndirectInPass);
            const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | (passIndirect ? static_cast<VkAccessFlags>(VK_ACCESS_INDIRECT_COMMAND_READ_BIT) : VkAccessFlags{0})};
            context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
            Recorder::CountBarriers(Recorder::CommandClass::Draw, 1);
            if (passIndirect) recorder->NotePassIndirectReads();
        }
        if (inputs.vertexInputBehindGpuWrite) {
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT);
            Recorder::CountBarriers(Recorder::CommandClass::Draw, 1);
        }
        block->pipeline->Begin(recording, *block->framebuffer, state.renderExtent, state, masked);
    }
    NoteHeapRecord(false, false);
    resources.RepeatPushBindings(commands, block->pipeline->Layout(), push, prepared->selected, upload->Handle(), uploadOffset, *recorder);
    pushDrawConstants(*block->pipeline, commands, state, draw, shaders, resources, nullptr, recipe->pushStages, 0, false, inputs.meshLayerStride, nullptr);
    noteDrawFaultToken(context, *recorder, commands, shaders, draw, static_cast<std::uint32_t>(AgcDriver::FaultStage::graphics), nullptr);
    std::uint32_t subdraw = 0;
    recordDrawCommands(context, commands, state, draw, inputs, nullptr, VK_NULL_HANDLE, 0, trace, subdraw);
    recorder->EndGpuTiming(Recorder::NoTiming);
    recorder->LeaveRenderPassOpen(block->passKey, drawTiming, block->fenced, passAccess);
    record.effects.keptOnce = true;
    const auto recorderId = recorder->Id();
    for (std::size_t index = 0; index < block->targets.size(); ++index) {
        auto* target = block->targets[index];
        const auto [slice, count] = block->slices[index];
        if (target->RenoteInPass(recorderId, recorder->RenderPassGeneration(), slice, count)) {
            ++record.effects.notesSkipped;
            continue;
        }
        target->MarkSliceDirty(slice, count);
        if (recorder->RenderPassOpen()) target->RememberPassNote(recorderId, recorder->RenderPassGeneration(), slice, count);
    }
    record.effects.hold = out.shared ? ownHold : DrawRecordHold::Held;
    record.effects.sharedHold = out.shared;
    record.effects.passHeld = true;
    if (recorder->RenderPassOpen() && GuestMemory::CollectEpoch() != 0) {
        auto& remembered = record.pass;
        remembered.valid = true;
        remembered.recorder = recorderId;
        remembered.generation = recorder->RenderPassGeneration();
        remembered.epoch = GuestMemory::CollectEpoch();
        remembered.proof = GuestMemory::ProofSerial();
        remembered.pending = StorageTexture::PendingSerial();
        remembered.depthSerial = DepthSurfacesSerial();
        remembered.depthReleased = DepthSurfacesReleased();
        remembered.versions.clear();
        for (const auto* target : block->targets) remembered.versions.push_back(target->UploadVersion());
        if (!out.shared && RuntimeSwitches::Set(RuntimeSwitch::DirectPassTargets)) {
            auto& kept = sharedRecordPass();
            kept.pass = remembered;
            kept.colors.assign(state.colors.begin(), state.colors.end());
            kept.depth = state.depth;
            kept.depthWrite = state.depthWrite;
            kept.stencilTest = state.stencilTest;
            kept.stencilWritten = writesStencilTarget(state);
            kept.depthCopySource = state.depthCopySource;
            kept.views = recipe->targetViews;
            kept.targets.assign(block->targets.begin(), block->targets.end());
        }
    } else {
        record.pass.valid = false;
    }
    timing.Mark("draw_and_resource_release");
    out.taken = true;
    out.result.recorded = true;
    return out;
}

namespace {

struct ExpressReadsStorage {};
struct ExpressAttachmentsStorage {};

std::atomic<std::uint64_t> leanExpressTaken{0};
std::atomic<std::uint64_t> leanExpressTouched{0};
std::atomic<std::uint64_t> leanExpressChecked{0};
std::atomic<std::uint64_t> leanExpressDiffering{0};

bool leanExpressVerifyDue() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_LEAN_EXPRESS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    static std::uint64_t ordinal = 0;
    return every != 0 && ++ordinal % every == 0;
}

void noteLeanExpressVerify(bool clear) {
    leanExpressChecked.fetch_add(1, std::memory_order_relaxed);
    if (clear) return;
    static std::atomic<std::uint32_t> printed{0};
    if (leanExpressDiffering.fetch_add(1, std::memory_order_relaxed) == 0 || printed.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[lean-express-verify] the buffer tests refuse a commit whose buffer ranges lie in memory no GPU write was marked for\n");
}

struct ExpressDepthChecked {
    std::uint64_t recorder = 0;
    std::uint64_t generation = 0;
    std::uint64_t address = 0;
    VkImageView view = VK_NULL_HANDLE;
    std::uint8_t applied = 0;
};
struct ExpressDepthStorage {};

std::uint8_t depthIntent(const State& state) {
    return static_cast<std::uint8_t>((state.depthWrite ? 1u : 0u) | (state.stencilTest ? 2u : 0u) | (writesStencilTarget(state) ? 4u : 0u));
}

bool depthIntentHeld(std::uint8_t own, std::uint8_t applied) {
    return (own & ~applied) == 0;
}

std::atomic<std::uint64_t> depthIntentChecked{0};
std::atomic<std::uint64_t> depthIntentStronger{0};
std::atomic<std::uint64_t> depthIntentFollowersStronger{0};
std::atomic<std::uint64_t> depthIntentViews{0};

void noteDepthIntent(std::uint8_t own, std::uint8_t applied, bool follower) {
    const auto checked = depthIntentChecked.fetch_add(1, std::memory_order_relaxed) + 1u;
    if (!depthIntentHeld(own, applied)) {
        depthIntentStronger.fetch_add(1, std::memory_order_relaxed);
        if (follower) depthIntentFollowersStronger.fetch_add(1, std::memory_order_relaxed);
    }
    if (checked % 65536u != 0u) return;
    const auto stronger = depthIntentStronger.load(std::memory_order_relaxed);
    std::fprintf(stderr, "[express-depth-intent] %llu checked, %llu stronger (%llu of them followers); %llu depth views called for a stronger intent in a stamped pass\n", static_cast<unsigned long long>(checked), static_cast<unsigned long long>(stronger), static_cast<unsigned long long>(depthIntentFollowersStronger.load(std::memory_order_relaxed)), static_cast<unsigned long long>(depthIntentViews.load(std::memory_order_relaxed)));
}
struct ExpressMultisampleChecked {
    const void* owner = nullptr;
    std::uint64_t recorder = 0;
    std::uint64_t generation = 0;
};
struct ExpressMultisampleMemo {
    std::array<ExpressMultisampleChecked, 8> checked{};
    std::size_t next = 0;
};
struct ExpressMultisampleStorage {};

enum class AppendRefusal : std::uint8_t { None, Prepared, Linear, Pass, Depth, Targets, InputShape, Imports, Writers, InputWrites, InputStores, Buffers, Multisample, Count };
constexpr std::array<const char*, static_cast<std::size_t>(AppendRefusal::Count)> AppendRefusalNames{"none", "not prepared", "linear target pending", "pass not open under the key", "depth", "targets", "input shape", "imports or mapping", "copied writers", "pending write over an input", "label or queued store over an input", "buffers", "multisample target not checked in this pass"};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(AppendRefusal::Count)> appendRefusals{};

void noteAppendRefusal(AppendRefusal refusal) {
    appendRefusals[static_cast<std::size_t>(refusal)].fetch_add(1, std::memory_order_relaxed);
}

std::atomic<std::uint64_t> passReadsChecked{0};
std::atomic<std::uint64_t> passReadsDiffering{0};
std::atomic<std::uint64_t> appendVerifyChecked{0};
std::atomic<std::uint64_t> appendVerifyDiffering{0};

bool appendVerifyDue() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_APPEND_COMMITS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    static std::uint64_t ordinal = 0;
    return every != 0 && ++ordinal % every == 0;
}

struct AppendClaim {
    bool armed = false;
    bool confirmed = false;
    ~AppendClaim() {
        if (!armed) return;
        const auto checked = appendVerifyChecked.fetch_add(1, std::memory_order_relaxed) + 1u;
        const auto differing = confirmed ? appendVerifyDiffering.load(std::memory_order_relaxed) : appendVerifyDiffering.fetch_add(1, std::memory_order_relaxed) + 1u;
        if (!confirmed && differing <= 16u) std::fprintf(stderr, "[append-commits-verify] a commit the append conditions accepted was refused, reproved, rechecked or refreshed by the full express commit\n");
        if (checked % 16384u == 0u) std::fprintf(stderr, "[append-commits-verify] %llu checked, %llu differing\n", static_cast<unsigned long long>(checked), static_cast<unsigned long long>(differing));
    }
};

std::atomic<std::uint64_t> appendCommitsTaken{0};
std::atomic<std::uint64_t> appendCommitsFull{0};

void noteAppendCommit(bool appended) {
    (appended ? appendCommitsTaken : appendCommitsFull).fetch_add(1, std::memory_order_relaxed);
    static std::uint64_t seen = 0;
    if ((++seen & 0xfffffu) != 0) return;
    const auto wide = [](std::atomic<std::uint64_t>& counter) { return static_cast<unsigned long long>(counter.exchange(0, std::memory_order_relaxed)); };
    std::string refused;
    for (std::size_t index = 1; index < appendRefusals.size(); ++index) {
        const auto count = appendRefusals[index].exchange(0, std::memory_order_relaxed);
        if (count != 0) refused += std::string(refused.empty() ? "" : ", ") + AppendRefusalNames[index] + " " + std::to_string(count);
    }
    std::fprintf(stderr, "[append-commits] %llu express commits appended to the open pass, %llu taken in full (not appended: %s)\n", wide(appendCommitsTaken), wide(appendCommitsFull), refused.empty() ? "none" : refused.c_str());
}

void noteLeanExpress(bool untouched) {
    (untouched ? leanExpressTaken : leanExpressTouched).fetch_add(1, std::memory_order_relaxed);
    static std::uint64_t seen = 0;
    if ((++seen & 0xfffffu) != 0) return;
    const auto wide = [](std::atomic<std::uint64_t>& counter) { return static_cast<unsigned long long>(counter.exchange(0, std::memory_order_relaxed)); };
    std::fprintf(stderr, "[lean-express] %llu express commits without buffer tests, %llu with them (a range in memory a GPU write was marked for); verify %llu checked, %llu differing\n", wide(leanExpressTaken), wide(leanExpressTouched), wide(leanExpressChecked), wide(leanExpressDiffering));
}

}

namespace {

std::array<std::atomic<std::uint64_t>, 8> expressTargetCauses{};

}

bool DrawBufferAliasesClear(std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint64_t indexAddress, std::uint64_t indexBytes) {
    try {
        CheckBufferAliases(shaders, target, indexAddress, indexBytes);
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

std::array<std::uint64_t, 8> ExpressTargetMissCauses() {
    std::array<std::uint64_t, 8> causes{};
    for (std::size_t index = 0; index < causes.size(); ++index) causes[index] = expressTargetCauses[index].load(std::memory_order_relaxed);
    return causes;
}

std::uint64_t ExpressVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_EXPRESS_COMMITS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

namespace {

std::uint32_t commitFeedParts() {
    static const std::uint32_t parts = [] {
        const char* text = std::getenv("APS5_COMMIT_FEED_PARTS");
        return text != nullptr ? static_cast<std::uint32_t>(std::strtoul(text, nullptr, 10)) : CommitFeedBlocks | CommitFeedCards | CommitFeedAhead;
    }();
    return parts;
}

constexpr std::size_t CommitFeedCounts = static_cast<std::size_t>(CommitFeedCount::Count);
std::array<std::atomic<std::uint64_t>, CommitFeedCounts> commitFeedTotals{};
std::atomic<std::uint64_t> commitFeedChecked{0};
std::atomic<std::uint64_t> commitFeedDiffering{0};
std::atomic<std::int64_t> commitFeedWindow{0};
std::atomic<std::uint32_t> commitFeedPrinted{0};

}

bool CommitFeedOn(std::uint32_t part) {
    return RuntimeSwitches::Set(RuntimeSwitch::CommitFeed) && (commitFeedParts() & part) != 0;
}

bool CommitFeedCensus() {
    static const bool census = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    return census;
}

std::uint64_t CommitFeedVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_COMMIT_FEED");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

void NoteCommitFeed(CommitFeedCount count, std::uint64_t amount) {
    auto& total = commitFeedTotals[static_cast<std::size_t>(count)];
    total.store(total.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

void NoteCommitFeedVerify(bool same, const char* what, const std::string& detail) {
    const auto checked = commitFeedChecked.fetch_add(1, std::memory_order_relaxed) + 1u;
    const auto differing = same ? commitFeedDiffering.load(std::memory_order_relaxed) : commitFeedDiffering.fetch_add(1, std::memory_order_relaxed) + 1u;
    if (!same) {
        const auto now = static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10);
        if (commitFeedWindow.exchange(now, std::memory_order_relaxed) != now) commitFeedPrinted.store(0, std::memory_order_relaxed);
        if (commitFeedPrinted.fetch_add(1, std::memory_order_relaxed) < 16u) std::fprintf(stderr, "[commit-feed-verify] %s differs: %s\n", what, detail.c_str());
    }
    if (checked % 4096u == 0u) std::fprintf(stderr, "[commit-feed-verify] %llu checked, %llu differing\n", static_cast<unsigned long long>(checked), static_cast<unsigned long long>(differing));
}

std::uint64_t CommitFeedTotal(CommitFeedCount count) {
    return commitFeedTotals[static_cast<std::size_t>(count)].load(std::memory_order_relaxed);
}

std::array<std::uint64_t, 2> CommitFeedVerifyTotals() {
    return {commitFeedChecked.load(std::memory_order_relaxed), commitFeedDiffering.load(std::memory_order_relaxed)};
}

std::string TakeCommitFeedText() {
    static std::mutex takeMutex;
    static std::array<std::uint64_t, CommitFeedCounts> reported{};
    static std::array<std::uint64_t, 2> verifyReported{};
    const std::lock_guard lock(takeMutex);
    std::array<unsigned long long, CommitFeedCounts> since{};
    bool any = false;
    for (std::size_t index = 0; index < CommitFeedCounts; ++index) {
        const auto now = commitFeedTotals[index].load(std::memory_order_relaxed);
        since[index] = static_cast<unsigned long long>(now - reported[index]);
        reported[index] = now;
        any = any || since[index] != 0;
    }
    const auto verify = CommitFeedVerifyTotals();
    const auto checked = static_cast<unsigned long long>(verify[0] - verifyReported[0]);
    const auto differing = static_cast<unsigned long long>(verify[1] - verifyReported[1]);
    verifyReported = verify;
    if (!any && checked == 0 && !RuntimeSwitches::Set(RuntimeSwitch::CommitFeed)) return {};
    const auto at = [&](CommitFeedCount count) { return since[static_cast<std::size_t>(count)]; };
    std::array<char, 2048> text{};
    std::snprintf(text.data(), text.size(), "fragment blocks kept over a commit without prepared buffers %llu, withdrawn %llu; zone cards used by followers %llu, by leaders %llu, for depth guards %llu, prepared buffers without a usable card %llu; lookahead %llu items noted, %llu without a hint, steps by level %llu, %llu, %llu, %llu, %llu, %llu; verify %llu checked, %llu differing; express leads refused: attachments not held %llu, more than 9 targets %llu, target versions or DCC addresses short %llu, pass not open %llu, pass not appendable %llu, multisample owner %llu; follows refused as not a direct record hit: not direct %llu, traced %llu, no recipe %llu, writers %llu, no buffer plan %llu, no inputs recipe %llu, no fragment %llu, no prepared buffers %llu, no prepared inputs %llu",
        at(CommitFeedCount::BlocksKept), at(CommitFeedCount::BlocksWithdrawn), at(CommitFeedCount::CardFollowerUses), at(CommitFeedCount::CardLeaderUses), at(CommitFeedCount::CardGuardUses), at(CommitFeedCount::HitsWithoutCard),
        at(CommitFeedCount::AheadEntries), at(CommitFeedCount::AheadNoHint), at(CommitFeedCount::AheadLevel0), at(CommitFeedCount::AheadLevel1), at(CommitFeedCount::AheadLevel2), at(CommitFeedCount::AheadLevel3), at(CommitFeedCount::AheadLevel4), at(CommitFeedCount::AheadLevel5),
        checked, differing,
        at(CommitFeedCount::LeadAttachments), at(CommitFeedCount::LeadTargets), at(CommitFeedCount::LeadShort), at(CommitFeedCount::LeadPassClosed), at(CommitFeedCount::LeadNotAppendable), at(CommitFeedCount::LeadMultisample),
        at(CommitFeedCount::NotDirectHit), at(CommitFeedCount::NotDirectTraced), at(CommitFeedCount::NotDirectRecipe), at(CommitFeedCount::NotDirectWriters), at(CommitFeedCount::NotDirectPlan), at(CommitFeedCount::NotDirectInputsRecipe), at(CommitFeedCount::NotDirectFragment), at(CommitFeedCount::NotDirectBuffers), at(CommitFeedCount::NotDirectInputs));
    return std::string(text.data());
}

namespace {

std::uint32_t followAheadParts() {
    static const std::uint32_t parts = [] {
        const char* text = std::getenv("APS5_FOLLOW_AHEAD_PARTS");
        return text != nullptr ? static_cast<std::uint32_t>(std::strtoul(text, nullptr, 10)) : FollowAheadGuards | FollowAheadRenotes;
    }();
    return parts;
}

constexpr std::size_t FollowAheadCounts = static_cast<std::size_t>(FollowAheadCount::Count);
std::array<std::atomic<std::uint64_t>, FollowAheadCounts> followAheadTotals{};
std::atomic<std::uint64_t> followAheadChecked{0};
std::atomic<std::uint64_t> followAheadDiffering{0};
std::atomic<std::int64_t> followAheadWindow{0};
std::atomic<std::uint32_t> followAheadPrinted{0};
std::uint64_t followAheadVerifyOrdinal = 0;

void noteFollowAhead(FollowAheadCount count, std::uint64_t amount = 1) {
    auto& total = followAheadTotals[static_cast<std::size_t>(count)];
    total.store(total.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

void noteFollowAheadVerify(bool same, const char* what, const std::string& detail) {
    const auto checked = followAheadChecked.fetch_add(1, std::memory_order_relaxed) + 1u;
    const auto differing = same ? followAheadDiffering.load(std::memory_order_relaxed) : followAheadDiffering.fetch_add(1, std::memory_order_relaxed) + 1u;
    if (!same) {
        const auto now = static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10);
        if (followAheadWindow.exchange(now, std::memory_order_relaxed) != now) followAheadPrinted.store(0, std::memory_order_relaxed);
        if (followAheadPrinted.fetch_add(1, std::memory_order_relaxed) < 16u) std::fprintf(stderr, "[follow-ahead-verify] %s differs: %s\n", what, detail.c_str());
    }
    if (checked % 4096u == 0u) std::fprintf(stderr, "[follow-ahead-verify] %llu checked, %llu differing\n", static_cast<unsigned long long>(checked), static_cast<unsigned long long>(differing));
}

bool followAheadVerifyDue() {
    const auto every = FollowAheadVerifyEvery();
    return every != 0 && ++followAheadVerifyOrdinal % every == 0;
}

struct GuardBoards {
    GuardSpanBoard::View depth;
    GuardSpanBoard::View multisample;
};

GuardAnswerMade openGuardBoards(GuardAnswer& answer, GuardBoards& views, std::uint64_t stamp, bool stamped) {
    if (RuntimeSwitches::Set(RuntimeSwitch::NoGuardSpans) || VerifyGuardSpans()) return GuardAnswerMade::SpansOff;
    if (DepthOverwriteProofsPending_nid_no_patch() != 0u) return GuardAnswerMade::ProofPending;
    answer.depth = DepthGuardsGeneration_nid_no_patch();
    answer.multisample = MultisampleGuardsGeneration();
    if (stamped && answer.depth != stamp) return GuardAnswerMade::StampMoved;
    if (!DepthGuardBoard().Open(answer.depth, views.depth) || !MultisampleGuardBoard().Open(answer.multisample, views.multisample)) return GuardAnswerMade::BoardStale;
    return GuardAnswerMade::Made;
}

GuardAnswerMade closeGuardBoards(const GuardAnswer& answer, const GuardBoards& views) {
    if (!DepthGuardBoard().Holds(views.depth) || !MultisampleGuardBoard().Holds(views.multisample)) return GuardAnswerMade::BoardStale;
    if (DepthGuardsGeneration_nid_no_patch() != answer.depth || MultisampleGuardsGeneration() != answer.multisample) return GuardAnswerMade::GenerationMoved;
    if (DepthOverwriteProofsPending_nid_no_patch() != 0u) return GuardAnswerMade::ProofPending;
    return GuardAnswerMade::Made;
}

bool depthBoardHit(const GuardBoards& views, std::uint64_t begin, std::uint64_t bytes) {
    return bytes != 0u && (begin > std::numeric_limits<std::uint64_t>::max() - bytes || DepthGuardBoard().Overlaps(views.depth, begin, bytes));
}

bool multisampleBoardHit(const GuardBoards& views, std::uint64_t begin, std::uint64_t bytes) {
    return bytes != 0u && MultisampleGuardBoard().Overlaps(views.multisample, begin, bytes);
}

std::string guardAnswerDetail(const char* what, std::uint64_t begin, std::uint64_t bytes, const GuardAnswer& answer) {
    char text[256];
    std::snprintf(text, sizeof(text), "%s 0x%llx+0x%llx, stamps depth %llu multisample %llu, live depth %llu multisample %llu", what, static_cast<unsigned long long>(begin), static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(answer.depth), static_cast<unsigned long long>(answer.multisample), static_cast<unsigned long long>(DepthGuardsGeneration_nid_no_patch()), static_cast<unsigned long long>(MultisampleGuardsGeneration()));
    return text;
}

std::uint8_t carriedGuards(const GuardAnswer& answer, std::uint8_t wanted, bool census) {
    if (!answer.valid || (answer.clear & wanted) == 0) {
        if (census) noteFollowAhead(FollowAheadCount::NoAnswer);
        return 0;
    }
    if (RuntimeSwitches::Set(RuntimeSwitch::NoGuardSpans)) {
        if (census) noteFollowAhead(FollowAheadCount::SpansOff);
        return 0;
    }
    if (DepthGuardsGeneration_nid_no_patch() != answer.depth) {
        if (census) noteFollowAhead(FollowAheadCount::DepthMoved);
        return 0;
    }
    if (MultisampleGuardsGeneration() != answer.multisample) {
        if (census) noteFollowAhead(FollowAheadCount::MultisampleMoved);
        return 0;
    }
    if (DepthOverwriteProofsPending_nid_no_patch() != 0u) {
        if (census) noteFollowAhead(FollowAheadCount::ProofPending);
        return 0;
    }
    return static_cast<std::uint8_t>(answer.clear & wanted);
}

enum class FollowAheadSite : std::uint8_t { Follower, Leader };

std::uint8_t carriedBufferGuards(const ShaderResources::PreparedBuffers& prepared, FollowAheadSite site) {
    if (!FollowAheadOn(FollowAheadGuards)) return 0;
    const bool census = FollowAheadCensus();
    const auto bits = carriedGuards(prepared.guardAnswer, GuardAnswer::DepthClear | GuardAnswer::MultisampleClear, census);
    if (bits == 0) return 0;
    if (followAheadVerifyDue()) {
        std::string detail;
        if ((bits & GuardAnswer::DepthClear) != 0) {
            for (const auto& item : prepared.selected) {
                if (item.begin == 0 || item.inPlace || !MayOverlapDepthGuards_nid_no_patch(item.begin, item.bytes)) continue;
                detail = guardAnswerDetail("depth", item.begin, item.bytes, prepared.guardAnswer);
                break;
            }
        }
        if (detail.empty() && (bits & GuardAnswer::MultisampleClear) != 0) {
            for (const auto& entry : prepared.moved) {
                if (!MayOverlapMultisampleGuards(entry.address, entry.size)) continue;
                detail = guardAnswerDetail("multisample", entry.address, entry.size, prepared.guardAnswer);
                break;
            }
        }
        noteFollowAheadVerify(detail.empty(), "guard answer", detail);
        if (!detail.empty()) {
            if (census) noteFollowAhead(FollowAheadCount::VerifyDiffers);
            return 0;
        }
    }
    if (census) {
        const bool follower = site == FollowAheadSite::Follower;
        if ((bits & GuardAnswer::DepthClear) != 0) noteFollowAhead(follower ? FollowAheadCount::FollowerDepth : FollowAheadCount::LeaderDepth);
        if ((bits & GuardAnswer::MultisampleClear) != 0) noteFollowAhead(follower ? FollowAheadCount::FollowerMultisample : FollowAheadCount::LeaderMultisample);
    }
    return bits;
}

bool carriedInputGuards(const PreparedInputs& prepared, FollowAheadSite site) {
    if (!FollowAheadOn(FollowAheadGuards)) return false;
    const bool census = FollowAheadCensus();
    if (carriedGuards(prepared.guardAnswer, GuardAnswer::InputsClear, census) == 0) return false;
    if (followAheadVerifyDue()) {
        std::string detail;
        for (const auto& range : prepared.ranges) {
            const auto bytes = static_cast<std::size_t>(range.end - range.begin);
            if (DepthReadGuardIdle_nid_no_patch(range.begin, bytes) && !MayOverlapMultisampleGuards(range.begin, bytes)) continue;
            detail = guardAnswerDetail("inputs", range.begin, bytes, prepared.guardAnswer);
            break;
        }
        noteFollowAheadVerify(detail.empty(), "guard answer", detail);
        if (!detail.empty()) {
            if (census) noteFollowAhead(FollowAheadCount::VerifyDiffers);
            return false;
        }
    }
    if (census) noteFollowAhead(site == FollowAheadSite::Follower ? FollowAheadCount::FollowerInputs : FollowAheadCount::LeaderInputs);
    return true;
}

}

std::uint32_t FollowAheadParts() {
    return followAheadParts();
}

bool FollowAheadOn(std::uint32_t part) {
    return RuntimeSwitches::Set(RuntimeSwitch::FollowAhead) && (followAheadParts() & part) != 0;
}

bool FollowAheadCensus() {
    static const bool census = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    return census;
}

std::uint64_t FollowAheadVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_FOLLOW_AHEAD");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

GuardAnswerNote AnswerPreparedGuards(ShaderResources::PreparedBuffers& prepared) {
    GuardAnswerNote note;
    auto& answer = prepared.guardAnswer;
    answer = {};
    if (!FollowAheadOn(FollowAheadGuards)) return note;
    GuardBoards views;
    note.made = openGuardBoards(answer, views, prepared.depthGuards, true);
    if (note.made != GuardAnswerMade::Made) return note;
    auto clear = static_cast<std::uint8_t>(GuardAnswer::DepthClear | GuardAnswer::MultisampleClear);
    for (const auto& item : prepared.selected) {
        if (item.begin == 0 || item.inPlace) continue;
        ++note.depthRanges;
        if ((clear & GuardAnswer::DepthClear) != 0 && depthBoardHit(views, item.begin, item.bytes)) clear = static_cast<std::uint8_t>(clear & ~GuardAnswer::DepthClear);
    }
    for (const auto& entry : prepared.moved) {
        ++note.multisampleRanges;
        if ((clear & GuardAnswer::MultisampleClear) != 0 && multisampleBoardHit(views, entry.address, entry.size)) clear = static_cast<std::uint8_t>(clear & ~GuardAnswer::MultisampleClear);
    }
    note.made = closeGuardBoards(answer, views);
    if (note.made != GuardAnswerMade::Made) return note;
    answer.clear = clear;
    answer.valid = true;
    note.clear = clear;
    return note;
}

GuardAnswerNote AnswerPreparedInputGuards(PreparedInputs& prepared) {
    GuardAnswerNote note;
    auto& answer = prepared.guardAnswer;
    answer = {};
    if (!FollowAheadOn(FollowAheadGuards)) return note;
    GuardBoards views;
    note.made = openGuardBoards(answer, views, 0, false);
    if (note.made != GuardAnswerMade::Made) return note;
    bool clear = true;
    for (const auto& range : prepared.ranges) {
        const auto bytes = range.end - range.begin;
        ++note.depthRanges;
        if (clear && (depthBoardHit(views, range.begin, bytes) || multisampleBoardHit(views, range.begin, bytes))) clear = false;
    }
    note.made = closeGuardBoards(answer, views);
    if (note.made != GuardAnswerMade::Made) return note;
    answer.clear = clear ? GuardAnswer::InputsClear : std::uint8_t{0};
    answer.valid = true;
    note.clear = answer.clear;
    return note;
}

std::uint64_t FollowAheadTotal(FollowAheadCount count) {
    return followAheadTotals[static_cast<std::size_t>(count)].load(std::memory_order_relaxed);
}

std::array<std::uint64_t, 2> FollowAheadVerifyTotals() {
    return {followAheadChecked.load(std::memory_order_relaxed), followAheadDiffering.load(std::memory_order_relaxed)};
}

std::string TakeFollowAheadText() {
    static std::mutex takeMutex;
    static std::array<std::uint64_t, FollowAheadCounts> reported{};
    static std::array<std::uint64_t, 2> verifyReported{};
    static std::array<std::uint64_t, 2> proofsReported{};
    static std::uint64_t presentsReported = 0;
    static std::uint64_t depthReported = 0;
    static std::uint64_t multisampleReported = 0;
    const std::lock_guard lock(takeMutex);
    std::array<unsigned long long, FollowAheadCounts> since{};
    bool any = false;
    for (std::size_t index = 0; index < FollowAheadCounts; ++index) {
        const auto now = followAheadTotals[index].load(std::memory_order_relaxed);
        since[index] = static_cast<unsigned long long>(now - reported[index]);
        reported[index] = now;
        any = any || since[index] != 0;
    }
    const auto verify = FollowAheadVerifyTotals();
    const auto checked = static_cast<unsigned long long>(verify[0] - verifyReported[0]);
    const auto differing = static_cast<unsigned long long>(verify[1] - verifyReported[1]);
    verifyReported = verify;
    const auto presentsNow = Recorder::Presents();
    const auto depthNow = DepthGuardsGeneration_nid_no_patch();
    const auto multisampleNow = MultisampleGuardsGeneration();
    const auto proofsNow = DepthOverwriteProofCounts();
    const auto presents = static_cast<unsigned long long>(presentsNow - presentsReported);
    const auto depthBumps = static_cast<unsigned long long>(depthNow - depthReported);
    const auto multisampleBumps = static_cast<unsigned long long>(multisampleNow - multisampleReported);
    const auto proofs = static_cast<unsigned long long>(proofsNow[0] - proofsReported[0]);
    const auto published = static_cast<unsigned long long>(proofsNow[1] - proofsReported[1]);
    presentsReported = presentsNow;
    depthReported = depthNow;
    multisampleReported = multisampleNow;
    proofsReported = proofsNow;
    if (!any && checked == 0 && !RuntimeSwitches::Set(RuntimeSwitch::FollowAhead)) return {};
    const auto at = [&](FollowAheadCount count) { return since[static_cast<std::size_t>(count)]; };
    const auto per = [](unsigned long long value, unsigned long long over) { return over == 0 ? 0.0 : static_cast<double>(value) / static_cast<double>(over); };
    const auto followers = at(FollowAheadCount::Followers);
    std::array<char, 2048> text{};
    std::snprintf(text.data(), text.size(), "committer answers used: followers' depth guards %llu, multisample guards %llu, input guards %llu, leaders' depth guards %llu, multisample guards %llu, input guards %llu; today's walk: no clear answer %llu, depth generation moved %llu, multisample generation moved %llu, overwrite proof pending %llu, guard spans off %llu, verify difference %llu; verify %llu checked, %llu differing; hold segments %llu with %llu followers (%.2f per segment), followers with the template of the follower before %llu (%.1f%%), unmoved zone card entries per follower %.2f, followers with queued stores %llu, followers whose target renotes a merge would take %llu; per present (%llu presents): depth guard generation moves %.2f, multisample %.2f, overwrite proofs %.2f of which published guards %.2f",
        at(FollowAheadCount::FollowerDepth), at(FollowAheadCount::FollowerMultisample), at(FollowAheadCount::FollowerInputs), at(FollowAheadCount::LeaderDepth), at(FollowAheadCount::LeaderMultisample), at(FollowAheadCount::LeaderInputs),
        at(FollowAheadCount::NoAnswer), at(FollowAheadCount::DepthMoved), at(FollowAheadCount::MultisampleMoved), at(FollowAheadCount::ProofPending), at(FollowAheadCount::SpansOff), at(FollowAheadCount::VerifyDiffers),
        checked, differing,
        at(FollowAheadCount::Segments), followers, per(followers, at(FollowAheadCount::Segments)), at(FollowAheadCount::SameTemplate), 100.0 * per(at(FollowAheadCount::SameTemplate), followers), per(at(FollowAheadCount::UnmovedEntries), followers), at(FollowAheadCount::QueuedStores), at(FollowAheadCount::MergeableRenotes),
        presents, per(depthBumps, presents), per(multisampleBumps, presents), per(proofs, presents), per(published, presents));
    return std::string(text.data());
}

namespace {

struct ExpressRunRanges {
    std::uint64_t recorder = 0;
    std::uint64_t first = 0;
    std::uint64_t second = 0;
    std::uint64_t third = 0;
    std::size_t count = 0;
    std::size_t next = 0;
    std::array<std::pair<std::uint64_t, std::uint64_t>, 8> ranges{};
};

struct ExpressRunCensus {
    std::uint64_t seen = 0;
    std::uint64_t followed = 0;
    std::uint64_t led = 0;
    std::uint64_t carried = 0;
    std::uint64_t notCarried = 0;
    std::uint64_t rangesSkipped = 0;
    std::uint64_t readsSkipped = 0;
    std::uint64_t followNs = 0;
    std::uint64_t followTimed = 0;
    std::uint64_t itemNs = 0;
    std::uint64_t itemTimed = 0;
    std::uint64_t refusedNs = 0;
    std::uint64_t refusedTimed = 0;
    std::uint64_t appended = 0;
    std::uint64_t full = 0;
    std::uint64_t appendedNs = 0;
    std::uint64_t appendedTimed = 0;
    std::uint64_t fullNs = 0;
    std::uint64_t fullTimed = 0;
    std::uint64_t holds = 0;
    std::uint64_t holdNs = 0;
    std::uint64_t carriedItems = 0;
    std::uint64_t kept = 0;
    std::uint64_t imageFollowers = 0;
    std::uint64_t shadowSkipped = 0;
    std::uint64_t staged = 0;
    std::array<std::uint64_t, 8> stageNs{};
    std::uint64_t refusingNs = 0;
    std::uint64_t refusingTimed = 0;
    std::uint64_t fallbackNs = 0;
    std::uint64_t fallbackTimed = 0;
    std::uint64_t unledNs = 0;
    std::uint64_t unledTimed = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(ExpressRunRefusal::Count)> refusals{};
    std::array<std::uint64_t, static_cast<std::size_t>(ExpressRunEnd::Count)> ends{};
    std::array<std::uint64_t, 6> lengths{};
    std::array<std::uint64_t, 6> windowAnswers{};
    std::uint64_t windowsClear = 0;
    std::uint64_t windowsLive = 0;
    std::uint64_t widerReproved = 0;
    std::uint64_t widerUploads = 0;
    std::uint64_t widerRefused = 0;
    std::uint64_t resumedTargets = 0;
    std::uint64_t resumedProofs = 0;
    std::uint64_t resumedMisses = 0;
    std::array<std::uint64_t, 4> ownTargets{};
    std::array<std::uint64_t, static_cast<std::size_t>(ShaderResources::ExpressMiss::Count)> templateMisses{};
    std::uint64_t templateReprovable = 0;
};

enum class WindowFact : std::uint8_t { Pending, Writes, Labels, DepthReads, DepthSpans, MultisampleSpans, Count };
constexpr std::array<const char*, static_cast<std::size_t>(WindowFact::Count)> WindowFactNames{"pending", "writes", "labels", "depth reads", "depth spans", "multisample spans"};

struct ExpressRunWindows {
    static constexpr std::uint64_t Shift = 16;
    static constexpr std::size_t Slots = 16;
    struct Slot {
        std::uint64_t window = ~std::uint64_t{0};
        std::uint8_t clear = 0;
        std::uint8_t live = 0;
    };
    bool valid = false;
    std::array<std::uint64_t, 9> key{};
    std::array<Slot, Slots> slots{};
};

struct ExpressResume {
    const DrawRecordCommit* record = nullptr;
    std::uint32_t ordinal = 0;
    bool targets = false;
    bool proved = false;
    bool upload = false;
    bool depthChecked = false;
    std::uint64_t depthGeneration = 0;
    DrawExpressMiss miss = DrawExpressMiss::None;
    std::uint32_t refreshed = 0;
    ShaderResources::ExpressMiss templateMiss = ShaderResources::ExpressMiss::None;
    std::optional<ShaderResources::CleanProof> cleanProof;
    DrawRecipeOutcome result;
};

struct ExpressRun {
    static constexpr std::size_t Targets = 9;
    std::uint32_t open = 0;
    bool led = false;
    bool expectAppend = false;
    bool dropped = false;
    bool itemsBetween = false;
    bool overItems = false;
    ExpressRunRefusal refused = ExpressRunRefusal::Count;
    ExpressRunEnd dropReason = ExpressRunEnd::NotQueued;
    std::uint64_t dropSerial = 0;
    std::uint64_t length = 0;
    std::uint32_t followOrdinal = 0;
    std::uint32_t leadOrdinal = 0;
    std::uint64_t recorder = 0;
    std::uint64_t batch = 0;
    std::uint64_t generation = 0;
    std::uint64_t passKey = 0;
    std::uint64_t switches = 0;
    const void* framebuffer = nullptr;
    std::uint64_t forget = 0;
    std::uint64_t imports = 0;
    std::uint64_t mapping = 0;
    std::uint64_t depthSerial = 0;
    std::uint64_t depthReleased = 0;
    bool depthChecked = false;
    std::uint64_t depthAddress = 0;
    VkImageView depthView = VK_NULL_HANDLE;
    std::uint8_t depthApplied = 0;
    std::uint8_t expectKind = 0;
    bool followDue = false;
    bool intentNoted = false;
    std::size_t targets = 0;
    std::array<const void*, Targets> owners{};
    std::array<std::uint64_t, Targets> versions{};
    std::array<std::uint64_t, Targets> dcc{};
    std::array<VkImage, Targets> images{};
    std::array<std::pair<VkImage, bool>, Targets> attachments{};
    ExpressRunRanges clear;
    ExpressRunRanges noted;
    ExpressRunRanges passReads;
    ExpressRunRanges shadowClear;
    ExpressRunRanges shadowLive;
    ExpressRunWindows windows;
    ExpressResume resume;
    ExpressRunCensus census;
    std::uint64_t wvAttachments = 0;
    bool segmentOpen = false;
    bool segmentRenoted = false;
    const void* segmentResources = nullptr;
    std::size_t segmentTargets = 0;
    std::array<std::pair<std::uint32_t, std::uint32_t>, Targets> segmentSlices{};
};

struct ExpressRunStorage {};

std::atomic<std::uint64_t> expressRunAttachments{0};

ExpressRun& expressRun() {
    return HostThreadLocal<ExpressRun, ExpressRunStorage>();
}

void noteFollowSegment(ExpressRun& run, const DrawExpress& express, const DrawExpressKeep& keep, const ShaderResources::PreparedBuffers& prepared, Recorder& recorder, std::size_t renoted) {
    noteFollowAhead(FollowAheadCount::Followers);
    std::size_t targets = 0;
    for (const auto& target : keep.targets) {
        if (target != nullptr) ++targets;
    }
    const bool renotedAll = targets != 0 && renoted == targets && express.slices.size() >= keep.targets.size() && keep.targets.size() <= ExpressRun::Targets;
    if (!run.segmentOpen) {
        run.segmentOpen = true;
        noteFollowAhead(FollowAheadCount::Segments);
    } else {
        if (run.segmentResources == static_cast<const void*>(keep.resources.get())) noteFollowAhead(FollowAheadCount::SameTemplate);
        bool merged = renotedAll && run.segmentRenoted && run.segmentTargets == keep.targets.size();
        for (std::size_t index = 0; merged && index < keep.targets.size(); ++index) {
            if (keep.targets[index] != nullptr && run.segmentSlices[index] != express.slices[index]) merged = false;
        }
        if (merged) noteFollowAhead(FollowAheadCount::MergeableRenotes);
    }
    std::uint64_t unmoved = 0;
    for (const auto& item : prepared.selected) {
        if (!item.moved) ++unmoved;
    }
    noteFollowAhead(FollowAheadCount::UnmovedEntries, unmoved);
    if (recorder.HasQueuedStores() || recorder.HasQueuedKeyStores()) noteFollowAhead(FollowAheadCount::QueuedStores);
    run.segmentResources = keep.resources.get();
    run.segmentRenoted = renotedAll;
    run.segmentTargets = keep.targets.size();
    for (std::size_t index = 0; renotedAll && index < keep.targets.size(); ++index) run.segmentSlices[index] = express.slices[index];
}

void keyRunRanges(ExpressRunRanges& memo, std::uint64_t recorder, std::uint64_t first, std::uint64_t second, std::uint64_t third) {
    if (memo.recorder == recorder && memo.first == first && memo.second == second && memo.third == third) return;
    memo.recorder = recorder;
    memo.first = first;
    memo.second = second;
    memo.third = third;
    memo.count = 0;
    memo.next = 0;
}

bool runRangesHold(const ExpressRunRanges& memo, std::uint64_t begin, std::uint64_t end, bool contained) {
    for (std::size_t index = 0; index < memo.count; ++index) {
        const auto& [first, last] = memo.ranges[index];
        if (contained ? first <= begin && end <= last : first == begin && last == end) return true;
    }
    return false;
}

void addRunRange(ExpressRunRanges& memo, std::uint64_t begin, std::uint64_t end) {
    memo.ranges[memo.next] = {begin, end};
    memo.next = (memo.next + 1) % memo.ranges.size();
    memo.count = std::min(memo.count + 1, memo.ranges.size());
}

constexpr std::array<const char*, static_cast<std::size_t>(ExpressRunRefusal::Count)> RunRefusalNames{"not a direct record hit", "verify sampled", "lock not held", "labels pending", "experiment switch", "device", "express state", "switch generation or modes", "objects", "another recorder", "pass generation moved", "another pass key", "own writes", "own written images", "pass not appendable", "serials", "linear target pending", "copied writers", "targets", "multisample target", "depth", "template not current", "prepared buffers", "prepared inputs", "read over a write of the pass", "pass not fenced", "pending write over an input", "label or queued store over an input", "buffers", "queued store over the template", "verify", "selection", "staging", "pass moved", "own targets", "own depth serials", "image over an attachment or a written image of the pass", "GPU-indirect draw", "moved buffers bound in place"};
static_assert(RunRefusalNames.back() != nullptr, "every follow refusal needs a name");
constexpr std::array<const char*, static_cast<std::size_t>(ExpressRunEnd::Count)> RunEndNames{"next item not queued", "next item not a draw", "cap", "lock waiter", "idle waiter", "refusal", "switch", "stop"};
std::atomic<std::uint64_t> runFollowChecked{0};
std::atomic<std::uint64_t> runFollowDiffering{0};

bool runFollowVerifyDue() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_APPEND_COMMITS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    static std::uint64_t ordinal = 0;
    return every != 0 && ++ordinal % every == 0;
}

std::array<std::atomic<std::uint64_t>, 3> widerFollowChecked{};
std::array<std::atomic<std::uint64_t>, 3> widerFollowDiffering{};

const char* widerContext(std::uint8_t wider) {
    return wider == 1u ? " (a reproved follower)" : wider == 2u ? " (a follower over a new target upload)" : wider == 3u ? " (a reproved follower over a new target upload)" : wider == 4u ? " (an express commit resumed after a wider follower's steps)" : "";
}

void noteRunFollowVerify(bool same, const char* why, const char* context = "", const char* kind = "", std::uint8_t wider = 0) {
    const auto checked = runFollowChecked.fetch_add(1, std::memory_order_relaxed) + 1u;
    const auto differing = same ? runFollowDiffering.load(std::memory_order_relaxed) : runFollowDiffering.fetch_add(1, std::memory_order_relaxed) + 1u;
    for (std::size_t bit = 0; bit < widerFollowChecked.size(); ++bit) {
        if (((wider >> bit) & 1u) == 0) continue;
        widerFollowChecked[bit].fetch_add(1, std::memory_order_relaxed);
        if (!same) widerFollowDiffering[bit].fetch_add(1, std::memory_order_relaxed);
    }
    if (!same && differing <= 16u) std::fprintf(stderr, "[run-follow-verify] a commit the follow conditions accepted is not appended by the express commit: %s%s%s%s\n", kind, why, context, widerContext(wider));
    if (checked % 4096u != 0u) return;
    const auto reproved = widerFollowChecked[0].load(std::memory_order_relaxed);
    const auto uploaded = widerFollowChecked[1].load(std::memory_order_relaxed);
    const auto resumed = widerFollowChecked[2].load(std::memory_order_relaxed);
    if (reproved == 0 && uploaded == 0 && resumed == 0) std::fprintf(stderr, "[run-follow-verify] %llu checked, %llu differing\n", static_cast<unsigned long long>(checked), static_cast<unsigned long long>(differing));
    else std::fprintf(stderr, "[run-follow-verify] %llu checked, %llu differing (reproved followers %llu checked, %llu differing; followers over a new target upload %llu checked, %llu differing; resumed express commits %llu checked, %llu differing)\n", static_cast<unsigned long long>(checked), static_cast<unsigned long long>(differing), static_cast<unsigned long long>(reproved), static_cast<unsigned long long>(widerFollowDiffering[0].load(std::memory_order_relaxed)), static_cast<unsigned long long>(uploaded), static_cast<unsigned long long>(widerFollowDiffering[1].load(std::memory_order_relaxed)), static_cast<unsigned long long>(resumed), static_cast<unsigned long long>(widerFollowDiffering[2].load(std::memory_order_relaxed)));
}

std::atomic<std::uint64_t> windowChecked{0};
std::atomic<std::uint64_t> windowDifferingTotal{0};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(WindowFact::Count)> windowDiffering{};

bool windowVerifyDue() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_APPEND_COMMITS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    static std::uint64_t ordinal = 0;
    return every != 0 && ++ordinal % every == 0;
}

void noteWindowVerify(WindowFact fact, bool same) {
    const auto checked = windowChecked.fetch_add(1, std::memory_order_relaxed) + 1u;
    const auto differing = same ? windowDifferingTotal.load(std::memory_order_relaxed) : windowDifferingTotal.fetch_add(1, std::memory_order_relaxed) + 1u;
    if (!same) windowDiffering[static_cast<std::size_t>(fact)].fetch_add(1, std::memory_order_relaxed);
    if (!same && differing <= 16u) std::fprintf(stderr, "[run-follow-verify] a window answered %s clear for a range the exact test does not find clear\n", WindowFactNames[static_cast<std::size_t>(fact)]);
    if (checked % 4096u != 0u) return;
    const auto count = [](WindowFact counted) { return static_cast<unsigned long long>(windowDiffering[static_cast<std::size_t>(counted)].load(std::memory_order_relaxed)); };
    std::fprintf(stderr, "[run-follow-verify] window answers %llu checked, %llu differing (pending %llu, writes %llu, labels %llu, depth reads %llu, depth spans %llu, multisample spans %llu)\n", static_cast<unsigned long long>(checked), static_cast<unsigned long long>(differing), count(WindowFact::Pending), count(WindowFact::Writes), count(WindowFact::Labels), count(WindowFact::DepthReads), count(WindowFact::DepthSpans), count(WindowFact::MultisampleSpans));
}

const char* runFollowContext(const ExpressRun& run, const DrawExpress* express) {
    const bool images = express != nullptr && !express->images.empty();
    return images ? (run.overItems ? " (a follower with sampled images, in a run carried across a non-draw item)" : " (a follower with sampled images)") : run.overItems ? " (in a run carried across a non-draw item)" : "";
}

void noteRunFollow(ExpressRun& run, bool followed) {
    auto& census = run.census;
    if (followed) ++census.followed;
    if ((++census.seen & 0xfffffu) != 0) return;
    const auto wide = [](std::uint64_t& counter) { return static_cast<unsigned long long>(std::exchange(counter, 0)); };
    const auto per = [](std::uint64_t& nanoseconds, std::uint64_t& timed) {
        const double value = timed == 0 ? 0.0 : static_cast<double>(nanoseconds) / static_cast<double>(timed);
        nanoseconds = 0;
        timed = 0;
        return value;
    };
    std::string refused;
    for (std::size_t index = 0; index < census.refusals.size(); ++index) {
        const auto count = std::exchange(census.refusals[index], 0);
        if (count != 0) refused += std::string(refused.empty() ? "" : ", ") + RunRefusalNames[index] + " " + std::to_string(count);
    }
    std::string ended;
    for (std::size_t index = 0; index < census.ends.size(); ++index) {
        const auto count = std::exchange(census.ends[index], 0);
        if (count != 0) ended += std::string(ended.empty() ? "" : ", ") + RunEndNames[index] + " " + std::to_string(count);
    }
    std::array<unsigned long long, 6> lengths{};
    for (std::size_t index = 0; index < lengths.size(); ++index) lengths[index] = wide(census.lengths[index]);
    const auto follows = wide(census.followed);
    const auto leads = wide(census.led);
    const auto carried = wide(census.carried);
    const auto notCarried = wide(census.notCarried);
    const auto rangesSkipped = wide(census.rangesSkipped);
    const auto readsSkipped = wide(census.readsSkipped);
    const auto follower = per(census.followNs, census.followTimed);
    const auto item = per(census.itemNs, census.itemTimed);
    const auto refusedAttempt = per(census.refusedNs, census.refusedTimed);
    const auto appendedCount = wide(census.appended);
    const auto fullCount = wide(census.full);
    const auto appendedCommit = per(census.appendedNs, census.appendedTimed);
    const auto fullCommit = per(census.fullNs, census.fullTimed);
    const auto holdCount = census.holds;
    const auto hold = per(census.holdNs, census.holds);
    const auto carriedItems = wide(census.carriedItems);
    const auto kept = wide(census.kept);
    const auto imageFollowers = wide(census.imageFollowers);
    const auto shadowSkipped = wide(census.shadowSkipped);
    const auto refusing = per(census.refusingNs, census.refusingTimed);
    const auto fallback = per(census.fallbackNs, census.fallbackTimed);
    const auto unled = per(census.unledNs, census.unledTimed);
    std::array<double, 8> stages{};
    for (std::size_t index = 0; index < stages.size(); ++index) stages[index] = census.staged == 0 ? 0.0 : static_cast<double>(std::exchange(census.stageNs[index], 0)) / static_cast<double>(census.staged);
    census.staged = 0;
    std::fprintf(stderr, "[run-follow] %llu express commits followed a leader, %llu led a run; not followed: %s; runs ended by: %s; run lengths 1: %llu, 2-3: %llu, 4-7: %llu, 8-15: %llu, 16-31: %llu, 32+: %llu; carried across a dropped hold %llu (across non-draw items %llu), not carried %llu; runs kept across a non-follower commit %llu; followers with sampled images %llu; input ranges taken by the run's clear ranges %llu, pending reads already noted in the batch %llu, moved ranges taken by the shadow windows %llu; ns per follower %.0f (%.0f with its wrapper), per refused follow attempt %.0f; non-follower express commits appended %llu at %.0f ns, not appended %llu at %.0f ns; ns per non-follower item after a refused follow attempt %.0f to refuse and %.0f to commit, without an attempt %.0f to commit; run holds %llu at %.0f ns to take; ns per follower stage (1 in 64): checks %.0f, template %.0f, pass and inputs %.0f, zones and stores %.0f, keep bound and selection %.0f, staging and guest buffers %.0f, notes and keeps %.0f, splice and target notes %.0f; 1 in 8 timed; trusted %llu\n", follows, leads, refused.empty() ? "none" : refused.c_str(), ended.empty() ? "none" : ended.c_str(), lengths[0], lengths[1], lengths[2], lengths[3], lengths[4], lengths[5], carried, carriedItems, notCarried, kept, imageFollowers, rangesSkipped, readsSkipped, shadowSkipped, follower, item, refusedAttempt, appendedCount, appendedCommit, fullCount, fullCommit, refusing, fallback, unled, static_cast<unsigned long long>(holdCount), hold, stages[0], stages[1], stages[2], stages[3], stages[4], stages[5], stages[6], stages[7], static_cast<unsigned long long>(TakeCommitterTrusted()));
    std::array<unsigned long long, static_cast<std::size_t>(WindowFact::Count)> answers{};
    for (std::size_t index = 0; index < answers.size(); ++index) answers[index] = wide(census.windowAnswers[index]);
    const auto windowsClear = wide(census.windowsClear);
    const auto windowsLive = wide(census.windowsLive);
    if (windowsClear != 0 || windowsLive != 0) std::fprintf(stderr, "[run-follow] ranges answered by windows: pending %llu, writes %llu, labels %llu, depth reads %llu, depth spans %llu, multisample spans %llu; windows proved clear %llu, live %llu\n", answers[0], answers[1], answers[2], answers[3], answers[4], answers[5], windowsClear, windowsLive);
    const auto widerReproved = wide(census.widerReproved);
    const auto widerUploads = wide(census.widerUploads);
    const auto widerRefused = wide(census.widerRefused);
    const auto resumedTargets = wide(census.resumedTargets);
    const auto resumedProofs = wide(census.resumedProofs);
    const auto resumedMisses = wide(census.resumedMisses);
    std::array<unsigned long long, 4> ownTargets{};
    for (std::size_t index = 0; index < ownTargets.size(); ++index) ownTargets[index] = wide(census.ownTargets[index]);
    std::string templateMisses;
    for (std::size_t index = 0; index < census.templateMisses.size(); ++index) {
        static constexpr std::array<const char*, static_cast<std::size_t>(ShaderResources::ExpressMiss::Count)> MissNames{"none", "not proved", "watch", "forget", "serials", "sources", "views", "pending CPU", "staging", "regions"};
        const auto count = std::exchange(census.templateMisses[index], 0);
        if (count != 0) templateMisses += std::string(templateMisses.empty() ? "" : ", ") + MissNames[index] + " " + std::to_string(count);
    }
    const auto reprovable = wide(census.templateReprovable);
    if (RuntimeSwitches::Set(RuntimeSwitch::FollowWider) || widerReproved != 0 || widerUploads != 0 || resumedTargets != 0 || resumedProofs != 0) std::fprintf(stderr, "[run-follow] wider follows: followers after an inline reprove %llu, after an inline target refresh %llu, refused after their side effects %llu; express commits resumed after the target refresh %llu, after the reprove %llu, with the follower's miss %llu; own-targets refusals: count or owner %llu, stale express upload version %llu, stale express DCC address %llu, with multisample owners %llu; template-not-current refusals: %s (reprovable by the express commit's rule %llu)\n", widerReproved, widerUploads, widerRefused, resumedTargets, resumedProofs, resumedMisses, ownTargets[0], ownTargets[1], ownTargets[2], ownTargets[3], templateMisses.empty() ? "none" : templateMisses.c_str(), reprovable);
}

void finishRun(ExpressRun& run, ExpressRunEnd reason) {
    run.dropped = false;
    run.itemsBetween = false;
    run.windows.valid = false;
    run.segmentOpen = false;
    if (!run.led) return;
    run.led = false;
    run.overItems = false;
    ++run.census.ends[static_cast<std::size_t>(reason)];
    std::size_t bucket = 0;
    for (auto length = run.length; length > 1 && bucket + 1 < run.census.lengths.size(); length >>= 1) ++bucket;
    ++run.census.lengths[bucket];
    run.length = 0;
}

bool runRefusalEnds(ExpressRunRefusal why) {
    switch (why) {
    case ExpressRunRefusal::Unlocked:
    case ExpressRunRefusal::Labels:
    case ExpressRunRefusal::Experiment:
    case ExpressRunRefusal::Device:
    case ExpressRunRefusal::Switches:
    case ExpressRunRefusal::Recorder:
    case ExpressRunRefusal::Generation:
    case ExpressRunRefusal::NotAppendable:
    case ExpressRunRefusal::Serials:
    case ExpressRunRefusal::Linear:
    case ExpressRunRefusal::Writers:
    case ExpressRunRefusal::Targets:
    case ExpressRunRefusal::PassMoved:
        return true;
    default:
        return false;
    }
}

void refuseRun(ExpressRun& run, ExpressRunRefusal why) {
    run.refused = why;
    run.segmentOpen = false;
    ++run.census.refusals[static_cast<std::size_t>(why)];
    if (runRefusalEnds(why)) finishRun(run, ExpressRunEnd::Refusal);
    noteRunFollow(run, false);
}

bool leadExpressRun(ExpressRun& run, Recorder& recorder, const DrawExpress& express, const DrawExpressKeep& keep, const State& state, const DrawRecipe& recipe, bool attachmentsHeld, std::uint8_t depthApplied) {
    const auto count = keep.targets.size();
    const bool depthChecked = recipe.hasDepth && state.depth.has_value();
    const auto depthAddress = depthChecked ? state.depth->address : 0u;
    const auto depthView = recipe.targetViews.empty() ? VK_NULL_HANDLE : recipe.targetViews.back();
    bool keeps = run.led && !run.dropped && run.recorder == recorder.Id() && run.batch == recorder.OpenSerial() && run.generation == recorder.RenderPassGeneration() && run.passKey == express.passKey && run.switches == express.switches && run.targets == count && run.depthChecked == depthChecked && run.depthAddress == depthAddress && run.depthView == depthView;
    for (std::size_t index = 0; keeps && index < count; ++index) {
        const auto* target = keep.targets[index].get();
        const void* owner = target != nullptr ? static_cast<const void*>(target) : index < keep.multisample.size() ? static_cast<const void*>(keep.multisample[index].get()) : nullptr;
        const VkImage image = target != nullptr ? target->Image() : index < express.multisampleImages.size() ? express.multisampleImages[index] : VK_NULL_HANDLE;
        keeps = owner == run.owners[index] && image == run.images[index];
    }
    if (!keeps) finishRun(run, ExpressRunEnd::Refusal);
    if (!keeps) run.wvAttachments = expressRunAttachments.fetch_add(1, std::memory_order_relaxed) + 1;
    const auto fail = [&run] {
        finishRun(run, ExpressRunEnd::Refusal);
        return false;
    };
    if (!attachmentsHeld || count > ExpressRun::Targets || express.targetVersions.size() < count || express.dccAddresses.size() < count || !recorder.RenderPassOpen() || recorder.AppendablePass(express.passKey) == VK_NULL_HANDLE) {
        if (CommitFeedCensus()) NoteCommitFeed(!attachmentsHeld ? CommitFeedCount::LeadAttachments : count > ExpressRun::Targets ? CommitFeedCount::LeadTargets : (express.targetVersions.size() < count || express.dccAddresses.size() < count) ? CommitFeedCount::LeadShort : !recorder.RenderPassOpen() ? CommitFeedCount::LeadPassClosed : CommitFeedCount::LeadNotAppendable);
        return fail();
    }
    run.recorder = recorder.Id();
    run.batch = recorder.OpenSerial();
    run.generation = recorder.RenderPassGeneration();
    run.passKey = express.passKey;
    run.switches = express.switches;
    run.framebuffer = keep.framebuffer.get();
    run.forget = GuestMemory::ForgetSerial();
    run.imports = HostImportSetGeneration();
    run.mapping = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    run.depthSerial = DepthSurfacesSerial();
    run.depthReleased = DepthSurfacesReleased();
    run.depthChecked = depthChecked;
    run.depthAddress = depthAddress;
    run.depthView = depthView;
    run.depthApplied = keeps ? static_cast<std::uint8_t>(run.depthApplied | depthApplied) : depthApplied;
    run.targets = count;
    for (std::size_t index = 0; index < count; ++index) {
        const auto* target = keep.targets[index].get();
        if (target == nullptr && (index >= keep.multisample.size() || keep.multisample[index] == nullptr || index >= express.multisampleImages.size())) {
            if (CommitFeedCensus()) NoteCommitFeed(CommitFeedCount::LeadMultisample);
            return fail();
        }
        run.owners[index] = target != nullptr ? static_cast<const void*>(target) : static_cast<const void*>(keep.multisample[index].get());
        run.versions[index] = target != nullptr ? target->UploadVersion() : 0u;
        run.dcc[index] = express.dccAddresses[index];
        run.images[index] = target != nullptr ? target->Image() : express.multisampleImages[index];
        run.attachments[index] = {run.images[index], true};
    }
    if (keeps) {
        ++run.census.kept;
        return true;
    }
    run.led = true;
    run.length = 1;
    ++run.census.led;
    return true;
}

void keyRunWindows(ExpressRun& run, const Recorder& recorder) {
    auto& windows = run.windows;
    const std::array<std::uint64_t, 9> key{recorder.Id(), recorder.WriteNoteCount(), StorageTexture::PendingSerial(), Recorder::QueuedLabelSerial(), DepthGuardsGeneration_nid_no_patch(), MultisampleGuardsGeneration(), DepthSurfacesSerial(), DepthSurfacesReleased(), RuntimeSwitches::Generation()};
    if (windows.valid && windows.key == key) return;
    windows.valid = true;
    windows.key = key;
    windows.slots.fill({});
}

void keyRunGuardWindows(ExpressRun& run, const Recorder& recorder) {
    const auto& key = run.windows.key;
    if (!run.windows.valid || key[4] != DepthGuardsGeneration_nid_no_patch() || key[5] != MultisampleGuardsGeneration() || key[6] != DepthSurfacesSerial() || key[8] != RuntimeSwitches::Generation()) keyRunWindows(run, recorder);
}

bool proveRunWindow(const Recorder& recorder, WindowFact fact, std::uint64_t address) {
    constexpr std::size_t bytes = std::size_t{1} << ExpressRunWindows::Shift;
    switch (fact) {
    case WindowFact::Pending: return PendingZones::Clear(address, bytes) || (!recorder.PendingWriteOverlaps(address, bytes) && !PendingStorageOverlaps(address, bytes, nullptr));
    case WindowFact::Writes: return PendingZones::Clear(address, bytes) || !recorder.PendingWriteOverlaps(address, bytes);
    case WindowFact::Labels: return !recorder.PendingLabelIn(address, bytes);
    case WindowFact::DepthReads: return DepthReadGuardIdle_nid_no_patch(address, bytes);
    case WindowFact::DepthSpans: return !MayOverlapDepthGuards_nid_no_patch(address, bytes);
    case WindowFact::MultisampleSpans: return !MayOverlapMultisampleGuards(address, bytes);
    default: return false;
    }
}

bool runWindowsClear(ExpressRun& run, const Recorder& recorder, WindowFact fact, std::uint64_t begin, std::uint64_t bytes) {
    constexpr auto top = std::numeric_limits<std::uint64_t>::max();
    if (!run.windows.valid || bytes == 0 || begin > top - bytes) return false;
    const auto first = begin >> ExpressRunWindows::Shift;
    const auto last = (begin + bytes - 1u) >> ExpressRunWindows::Shift;
    if (last - first > 1u || last >= (top >> ExpressRunWindows::Shift)) return false;
    const auto bit = static_cast<std::uint8_t>(1u << static_cast<unsigned>(fact));
    for (auto window = first; window <= last; ++window) {
        auto& slot = run.windows.slots[window & (ExpressRunWindows::Slots - 1u)];
        if (slot.window != window) slot = {window, 0, 0};
        if ((slot.clear & bit) != 0) continue;
        if ((slot.live & bit) != 0) return false;
        if (!proveRunWindow(recorder, fact, window << ExpressRunWindows::Shift)) {
            slot.live |= bit;
            ++run.census.windowsLive;
            return false;
        }
        slot.clear |= bit;
        ++run.census.windowsClear;
    }
    ++run.census.windowAnswers[static_cast<std::size_t>(fact)];
    return true;
}

std::optional<bool> guardsByValidation(const Recorder& recorder, std::uint64_t address, std::uint64_t bytes, std::uint16_t fact, WriteValidation::Test test) {
    const auto answer = WriteValidation::CommitterClear(recorder, address, bytes, fact, test);
    if (answer == WriteValidation::Answer::Today) return std::nullopt;
    if (answer == WriteValidation::Answer::Proved || !WriteValidation::VerifyDue(test)) return false;
    const auto size = static_cast<std::size_t>(bytes);
    const bool depth = fact == WriteValidation::Fact::DepthSpans;
    const bool overlaps = depth ? MayOverlapDepthGuards_nid_no_patch(address, size) : MayOverlapMultisampleGuards(address, size);
    WriteValidation::NoteVerify(test, overlaps, depth ? "a selected buffer overlaps a depth guard" : "a moved buffer overlaps a multisample guard");
    return overlaps;
}

struct ValidationGuards {
    const Recorder& recorder;
    WriteValidation::Test test;
};

bool validationDepthGuards(void* tests, std::uint64_t address, std::uint64_t bytes) {
    const auto& guards = *static_cast<const ValidationGuards*>(tests);
    if (const auto answered = guardsByValidation(guards.recorder, address, bytes, WriteValidation::Fact::DepthSpans, guards.test); answered.has_value()) return *answered;
    return MayOverlapDepthGuards_nid_no_patch(address, static_cast<std::size_t>(bytes));
}

bool validationMultisampleGuards(void* tests, std::uint64_t address, std::uint64_t bytes) {
    const auto& guards = *static_cast<const ValidationGuards*>(tests);
    if (const auto answered = guardsByValidation(guards.recorder, address, bytes, WriteValidation::Fact::MultisampleSpans, guards.test); answered.has_value()) return *answered;
    return MayOverlapMultisampleGuards(address, static_cast<std::size_t>(bytes));
}

bool validationWrites(void* tests, std::uint64_t address, std::uint64_t bytes) {
    const auto& guards = *static_cast<const ValidationGuards*>(tests);
    const auto size = static_cast<std::size_t>(bytes);
    const auto answer = WriteValidation::CommitterClear(guards.recorder, address, bytes, WriteValidation::Fact::Writes, guards.test);
    if (answer == WriteValidation::Answer::Today) return guards.recorder.PendingWriteOverlaps(address, size);
    if (answer == WriteValidation::Answer::Proved || !WriteValidation::VerifyDue(guards.test)) return false;
    const bool overlaps = guards.recorder.PendingWriteOverlaps(address, size);
    WriteValidation::NoteVerify(guards.test, overlaps, "an unmoved selected buffer has a pending write");
    return overlaps;
}

struct FollowGuards {
    ExpressRun& run;
    const Recorder& recorder;
    bool validation = false;
};

bool followDepthGuards(void* tests, std::uint64_t address, std::uint64_t bytes) {
    auto& guards = *static_cast<FollowGuards*>(tests);
    const auto size = static_cast<std::size_t>(bytes);
    if (guards.validation) {
        if (const auto answered = guardsByValidation(guards.recorder, address, bytes, WriteValidation::Fact::DepthSpans, WriteValidation::Test::FollowerSelection); answered.has_value()) return *answered;
    }
    if (!runWindowsClear(guards.run, guards.recorder, WindowFact::DepthSpans, address, bytes)) return MayOverlapDepthGuards_nid_no_patch(address, size);
    if (!windowVerifyDue()) return false;
    const bool overlaps = MayOverlapDepthGuards_nid_no_patch(address, size);
    noteWindowVerify(WindowFact::DepthSpans, !overlaps);
    return overlaps;
}

bool followMultisampleGuards(void* tests, std::uint64_t address, std::uint64_t bytes) {
    auto& guards = *static_cast<FollowGuards*>(tests);
    const auto size = static_cast<std::size_t>(bytes);
    if (guards.validation) {
        if (const auto answered = guardsByValidation(guards.recorder, address, bytes, WriteValidation::Fact::MultisampleSpans, WriteValidation::Test::FollowerSelection); answered.has_value()) return *answered;
    }
    if (!runWindowsClear(guards.run, guards.recorder, WindowFact::MultisampleSpans, address, bytes)) return MayOverlapMultisampleGuards(address, size);
    if (!windowVerifyDue()) return false;
    const bool overlaps = MayOverlapMultisampleGuards(address, size);
    noteWindowVerify(WindowFact::MultisampleSpans, !overlaps);
    return overlaps;
}

struct FollowZones {
    ExpressRun& run;
    const Recorder& recorder;
    bool windows = false;
    bool validation = false;
};

bool followZoneToday(FollowZones& zones, std::uint64_t begin, std::size_t bytes, bool moved) {
    auto& run = zones.run;
    const auto end = begin + bytes;
    const auto pendingOver = [&] { return !PendingZones::Clear(begin, bytes) && (zones.recorder.PendingWriteOverlaps(begin, bytes) || (moved && PendingStorageOverlaps(begin, bytes, nullptr))); };
    bool clear = runRangesHold(run.clear, begin, end, true);
    if (!clear && zones.windows && runWindowsClear(run, zones.recorder, moved ? WindowFact::Pending : WindowFact::Writes, begin, bytes)) {
        clear = true;
        if (windowVerifyDue()) {
            clear = !pendingOver();
            noteWindowVerify(moved ? WindowFact::Pending : WindowFact::Writes, clear);
        }
    }
    if (!clear && pendingOver()) return false;
    if (!moved || bytes == 0) return true;
    constexpr std::uint64_t Window = std::uint64_t{1} << 20u;
    const auto first = begin & ~(Window - 1u);
    const auto last = end > std::numeric_limits<std::uint64_t>::max() - (Window - 1u) ? std::numeric_limits<std::uint64_t>::max() : (end + Window - 1u) & ~(Window - 1u);
    const auto serial = ShadowLiveSerial();
    keyRunRanges(run.shadowClear, 0, serial, 0, 0);
    keyRunRanges(run.shadowLive, 0, serial, 0, 0);
    if (runRangesHold(run.shadowClear, begin, end, true)) {
        ++run.census.shadowSkipped;
        return true;
    }
    if (!runRangesHold(run.shadowLive, first, last, false)) {
        if (!AnyShadowedOverlaps(first, static_cast<std::size_t>(last - first))) {
            addRunRange(run.shadowClear, first, last);
            return true;
        }
        addRunRange(run.shadowLive, first, last);
    }
    return !AnyShadowedOverlaps(begin, bytes);
}

bool followZoneClear(void* tests, std::uint64_t begin, std::size_t bytes, bool moved) {
    auto& zones = *static_cast<FollowZones*>(tests);
    if (zones.validation && bytes != 0) {
        constexpr auto test = WriteValidation::Test::FollowerZones;
        if (const auto answer = WriteValidation::CommitterClear(zones.recorder, begin, bytes, moved ? WriteValidation::Fact::Moved : WriteValidation::Fact::Writes, test); answer != WriteValidation::Answer::Today) {
            if (answer == WriteValidation::Answer::Proved || !WriteValidation::VerifyDue(test)) return true;
            const bool held = followZoneToday(zones, begin, bytes, moved);
            WriteValidation::NoteVerify(test, !held, "a template buffer has a pending write, a pending storage image or a live shadow");
            return held;
        }
    }
    return followZoneToday(zones, begin, bytes, moved);
}

constexpr std::size_t ExpressTargetHeld = 8;

std::size_t refreshExpressTarget(const Context& context, const State& state, DrawExpress& express, const std::shared_ptr<StorageTexture>& target, std::size_t index, Recorder& recorder, std::uint32_t& refreshed) {
    if (index >= state.colors.size() || !target->Cached()) return 2;
    const auto& color = state.colors[index];
    const auto dcc = SurfaceForTarget(color).dccAddress;
    if (dcc != express.dccAddresses[index] || !StorageImageServesKeys(*target, dcc)) return 3;
    const bool follows = RuntimeSwitches::Set(RuntimeSwitch::ExpressTargetUploads);
    const bool uploaded = follows && target->UploadVersion() != express.targetVersions[index];
    if (uploaded || !recorder.RenderPassOpen() || !target->NotedInPass(recorder.Id(), recorder.RenderPassGeneration())) {
        if (!StorageImageCached(context, target.get())) return 4;
        if (color.fmaskAddress != 0) GuardOneFragmentFmask(context, color);
        DrawOutcome outcome;
        const auto resident = refreshResidentTarget(context, state, color, outcome, false, [&] {
            if (color.arrayLastSlice != 0) target->RefreshLayers(color.arraySlice, 1);
            else target->Refresh();
            return target;
        });
        if (resident.get() != target.get()) return 5;
        ++refreshed;
    }
    if (target->UploadVersion() != express.targetVersions[index]) {
        if (!follows) return 6;
        express.targetVersions[index] = target->UploadVersion();
        expressTargetCauses[7].fetch_add(1, std::memory_order_relaxed);
    }
    return ExpressTargetHeld;
}

bool followWiderAllowed(const ExpressRun& run, const DrawExpressKeep& keep, const Pm4::DrawParameters& draw, const DrawFragment& fragment) {
    if (run.followDue || !RuntimeSwitches::Set(RuntimeSwitch::AppendCommits) || RuntimeSwitches::Set(RuntimeSwitch::XpAppendCommits) || RuntimeSwitches::Set(RuntimeSwitch::XpExpressFloor) || RuntimeSwitches::Set(RuntimeSwitch::DirectRepeat) || !RuntimeSwitches::Set(RuntimeSwitch::CommitFragments) || !RuntimeSwitches::Set(RuntimeSwitch::ExpressCommits)) return false;
    if (!RecordDraws() || DumpTargetLimit() != 0 || repeatExcludedModes() || !fragmentModes() || fragment.indirect != draw.indirect.has_value()) return false;
    return std::ranges::none_of(keep.targets, [](const std::shared_ptr<StorageTexture>& target) { return target == nullptr; }) && std::ranges::none_of(keep.multisample, [](const std::shared_ptr<void>& owner) { return owner != nullptr; });
}

}

ExpressOutcome ExpressCommit(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, DrawRecordCommit& record, const ShaderResources::PreparedBuffers* prepared, PreparedInputs* preparedInputs, DrawFragment* fragment, bool verifyDue) {
    if (const auto hook = expressHook.load(std::memory_order_acquire); hook != nullptr) hook(record);
    ExpressOutcome out;
    auto& run = expressRun();
    if (run.dropped) finishRun(run, run.dropReason);
    const bool expectAppend = std::exchange(run.expectAppend, false);
    const std::uint8_t pendingKind = std::exchange(run.expectKind, std::uint8_t{0});
    const std::uint8_t followKind = expectAppend ? pendingKind : std::uint8_t{0};
    ExpressResume* resume = run.resume.record == &record && run.resume.ordinal == run.followOrdinal ? &run.resume : nullptr;
    run.resume.record = nullptr;
    bool sideEffectsBegun = false;
    bool sideEffectsMoved = false;
    const bool intentOn = RuntimeSwitches::Set(RuntimeSwitch::ExpressDepthIntent);
    const std::uint8_t ownIntent = intentOn ? depthIntent(state) : 0u;
    bool intentNoted = std::exchange(run.intentNoted, false);
    std::uint8_t conditionsApplied = 0;
    std::uint8_t depthApplied = 0;
    const bool following = run.open != 0 && RuntimeSwitches::Set(RuntimeSwitch::RunFollow);
    const bool leadTimed = following && !expectAppend && (++run.leadOrdinal & 7u) == 0;
    const auto leadBegan = leadTimed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    bool validated = false;
    bool validatedSplit = false;
    std::uint64_t validatedAt = 0;
    const auto noteValidated = [&](std::uint64_t generation) {
        if (!validated) {
            validated = true;
            validatedAt = generation;
        } else if (validatedAt != generation) {
            validatedSplit = true;
        }
    };
    AppendRefusal followSeen = AppendRefusal::Count;
    bool followVerifyOpen = expectAppend && !verifyDue;
    const auto miss = [&](DrawExpressMiss why) {
        if (std::exchange(followVerifyOpen, false)) {
            if (followKind != 0 && sideEffectsBegun) noteRunFollowVerify(true, DrawExpressMissNames[static_cast<std::size_t>(why)], runFollowContext(run, record.express.get()), "the express commit returned after the follower's side effects on ", followKind);
            else noteRunFollowVerify(false, DrawExpressMissNames[static_cast<std::size_t>(why)], runFollowContext(run, record.express.get()), "the express commit returned before its pass start on ", followKind);
        }
        out.miss = why;
        return out;
    };
    const auto targetMiss = [&](std::size_t cause) {
        expressTargetCauses[cause].fetch_add(1, std::memory_order_relaxed);
        return miss(DrawExpressMiss::Targets);
    };
    auto* express = record.express.get();
    if (express == nullptr || !express->valid || express->keep == nullptr) return miss(DrawExpressMiss::NoState);
    auto* recorder = Recorder::Active();
    const auto* recipe = record.recipe.get();
    using IndirectRule = Pm4::DrawParameters::IndirectDraw::Rule;
    const bool indirectExpress = draw.indirect && RuntimeSwitches::Set(RuntimeSwitch::DirectGpuIndirect) && RuntimeSwitches::Set(RuntimeSwitch::ExpressGpuIndirect) && RuntimeSwitches::Set(RuntimeSwitch::IndirectInPass) && draw.indirect->vertexRule == IndirectRule::InPlace && draw.indirect->instanceRule == IndirectRule::InPlace && preparedInputs != nullptr && preparedInputs->indirectRanges == (draw.indirect->countIndirect ? 2u : 1u);
    if (recipe == nullptr || express->recipe != recipe || express->state != &state || express->switches != RuntimeSwitches::Generation() || recorder == nullptr || context.recorder != recorder || express->device != context.device || (draw.indirect && !indirectExpress) || !RecordDraws() || DumpTargetLimit() != 0 || recordChecksQueued() || repeatExcludedModes() || !fragmentModes() || record.expressRefused) return miss(DrawExpressMiss::Shape);
    auto& keep = *express->keep;
    if (fragment == nullptr || fragment->bindings == nullptr || fragment->device != context.device || fragment->resources.get() != keep.resources.get() || fragment->pipeline.get() != keep.pipeline.get() || fragment->indirect != draw.indirect.has_value()) return miss(DrawExpressMiss::Objects);
    auto& resources = *keep.resources;
    const bool trusted = RuntimeSwitches::Set(RuntimeSwitch::XpTrustCommitter);
    const bool audit = !trusted && RuntimeSwitches::Set(RuntimeSwitch::TrustAudit);
    const bool validation = WriteValidation::Session() != 0;
    const auto inPlaceValidation = validation ? std::optional<WriteValidation::Test>{WriteValidation::Test::LeaderInPlace} : std::optional<WriteValidation::Test>{};
    auto templateCheck = audit ? resources.TrustCheck(CommitterTrustFamily::Template, CommitterTrustSite::LeaderTemplate) : CommitterTrustCheck{};
    auto buffersCheck = audit ? resources.TrustCheck(CommitterTrustFamily::Buffers, CommitterTrustSite::LeaderBuffers) : CommitterTrustCheck{};
    auto inputsCheck = audit ? resources.TrustCheck(CommitterTrustFamily::Inputs, CommitterTrustSite::LeaderInputs) : CommitterTrustCheck{};
    const auto preparedMapped = [&] { return (prepared->forget & 1u) == 0 && GuestMemory::ForgetSerial() == prepared->forget; };
    const auto templateMiss = resume != nullptr ? resume->templateMiss : trusted ? resources.ExpressCurrentTrusted() : audit ? resources.ExpressCurrentAudited(templateCheck, false) : resources.ExpressCurrent();
    if (validation && resume == nullptr) WriteValidation::Note(WriteValidation::Test::TemplateCurrent, WriteValidation::Outcome::Kept);
    const bool reprove = resume != nullptr ? templateMiss != ShaderResources::ExpressMiss::None : templateMiss != ShaderResources::ExpressMiss::None && templateMiss != ShaderResources::ExpressMiss::Regions && (templateMiss != ShaderResources::ExpressMiss::NotProved || (RuntimeSwitches::Set(RuntimeSwitch::ExpressKeepState) && resources.ExpressWatchable()));
    if (templateMiss != ShaderResources::ExpressMiss::None && !reprove) return miss(DrawExpressMiss::Template);
    const bool appendOn = RuntimeSwitches::Set(RuntimeSwitch::XpAppendCommits);
    if (appendOn && !verifyDue && !draw.indirect && templateMiss == ShaderResources::ExpressMiss::None && prepared != nullptr && !prepared->inPlace && preparedInputs != nullptr && !preparedInputs->mirrored && !record.verifyBuffers && !record.verifyInputs && prepared->plan != nullptr && prepared->plan == resources.DirectBufferPlan(shaders) && !StorageTexture::DrawnLinearPending() && (!express->hasDepth || (DepthSurfacesSerial() == express->depthSerial && DepthSurfacesReleased() == express->depthReleased))) {
        if (const auto commands = recorder->AppendablePass(express->passKey); commands != VK_NULL_HANDLE) {
            if (auto* stream = CommandStream::FromHandle(commands); stream != nullptr) {
                bool staged = true;
                try {
                    resources.ExpressStage(*recorder);
                } catch (const std::exception&) {
                    staged = false;
                }
                if (staged && resources.FragmentSelectionHolds(*recorder, prepared->moved, *prepared, nullptr, true) == ShaderResources::PreparedBuffers::Selection::Taken && recorder->AppendablePass(express->passKey) == commands) {
                    if (resources.ChecksFaults() && fragment->faultGeneration != resources.FaultGeneration()) resources.RewriteFragmentFaults(*fragment->bindings);
                    resources.RequireGuestBuffers();
                    if (recipe->hasDepth && writesDepthTarget(state)) NoteDepthSurfaceWrite(context, *state.depth);
                    if (const auto present = Recorder::Presents(); record.touchedPresent != present) {
                        if (resources.TouchedPresent() != present) {
                            SharedResourceCache().Touch(recipe->key);
                            resources.NoteTouchedPresent(present);
                        }
                        record.touchedPresent = present;
                    }
                    keepExpressBindings(*recorder, fragment->bindings);
                    if (express->keptRecorder != recorder->Id() || express->keptBatch != recorder->OpenSerial()) {
                        recorder->Keep(express->keep);
                        for (const auto& owner : keep.multisample) {
                            if (owner != nullptr) KeepMultisampleTarget_nid_no_patch(owner);
                        }
                        express->keptRecorder = recorder->Id();
                        express->keptBatch = recorder->OpenSerial();
                    }
                    record.effects = {};
                    record.access.generation = recorder->RenderPassGeneration();
                    recorder->NoteSampledDraw(commands);
                    stream->Splice(std::move(fragment->commands));
                    recorder->GraphicsState().Invalidate();
                    NoteGuestWorkIn(context, commands);
                    record.effects.fragment.spliced = true;
                    record.effects.fragment.offered = true;
                    record.effects.inputs = DrawRecordInputs::Used;
                    record.effects.keptOnce = true;
                    record.effects.proved = true;
                    record.effects.cleanProof = ShaderResources::CleanProof::Skipped;
                    record.effects.hold = DrawRecordHold::Held;
                    record.effects.passHeld = true;
                    record.effects.buffers = DrawRecordBuffers::Used;
                    record.effects.pushed = fragment->pushing;
                    record.pass.valid = false;
                    if (resources.ChecksFaults() && resources.TakeFaultCheck(recorder->Id(), recorder->OpenSerial())) recorder->OnComplete([kept = keep.resources] { kept->CheckFaults(); });
                    noteAppendCommit(true);
                    out.taken = true;
                    out.result.recorded = true;
                    return out;
                }
            }
        }
    }
    if (appendOn) noteAppendCommit(false);
    struct FloorPass {
        std::uint64_t recorder = 0;
        std::uint64_t generation = 0;
        const void* framebuffer = nullptr;
    };
    thread_local FloorPass floorPass;
    const bool atFloor = RuntimeSwitches::Set(RuntimeSwitch::XpExpressFloor);
    if (indirectExpress && atFloor) return miss(DrawExpressMiss::Shape);
    AppendClaim appendClaim;
    bool appended = false;
    bool inPlaceBehind = false;
    bool inPlaceRefused = false;
    std::uint64_t conditionsRecorder = 0;
    std::uint64_t conditionsGeneration = 0;
    const auto conditions = [&](bool byRun, bool ignoreVersions) {
        const auto recorderId = conditionsRecorder;
        const auto generation = conditionsGeneration;
        AppendRefusal refusal = AppendRefusal::None;
        if (prepared == nullptr || preparedInputs == nullptr || record.verifyBuffers || record.verifyInputs || !RuntimeSwitches::Set(RuntimeSwitch::DirectBuffers) || !RuntimeSwitches::Set(RuntimeSwitch::DirectInputs) || prepared->plan == nullptr || prepared->plan != resources.DirectBufferPlan(shaders) || context.recorder == nullptr || !preparedInputsUsable(context, *preparedInputs)) refusal = AppendRefusal::Prepared;
        else if (StorageTexture::DrawnLinearPending()) refusal = AppendRefusal::Linear;
        else if (recorder->AppendablePass(express->passKey) == VK_NULL_HANDLE || (draw.indirect && !recorder->PassIndirectReads())) refusal = AppendRefusal::Pass;
        else if (express->hasDepth && (DepthSurfacesSerial() != express->depthSerial || DepthSurfacesReleased() != express->depthReleased)) refusal = AppendRefusal::Depth;
        for (std::size_t index = 0; refusal == AppendRefusal::None && index < keep.targets.size(); ++index) {
            const auto& target = keep.targets[index];
            if (target == nullptr) {
                const auto* owner = index < keep.multisample.size() ? keep.multisample[index].get() : nullptr;
                bool known = owner != nullptr && index < state.colors.size() && index < express->multisampleImages.size() && index < recipe->targetViews.size();
                if (known && byRun) {
                    known = run.recorder == recorderId && run.generation == generation && index < run.targets && run.owners[index] == owner && run.images[index] == express->multisampleImages[index];
                } else if (known) {
                    const auto& memo = HostThreadLocal<ExpressMultisampleMemo, ExpressMultisampleStorage>();
                    known = std::ranges::any_of(memo.checked, [&](const ExpressMultisampleChecked& entry) { return entry.owner == owner && entry.recorder == recorderId && entry.generation == generation; });
                }
                if (!known) {
                    refusal = AppendRefusal::Multisample;
                    break;
                }
                continue;
            }
            if (index >= state.colors.size() || !target->Cached() || !target->NotedInPass(recorderId, generation) || (!ignoreVersions && target->UploadVersion() != express->targetVersions[index])) {
                refusal = AppendRefusal::Targets;
                break;
            }
            const auto dcc = SurfaceForTarget(state.colors[index]).dccAddress;
            if (dcc != express->dccAddresses[index] || !StorageImageServesKeys(*target, dcc)) refusal = AppendRefusal::Targets;
        }
        if (refusal == AppendRefusal::None && recipe->hasDepth && state.depth.has_value()) {
            const auto expected = recipe->targetViews.empty() ? VK_NULL_HANDLE : recipe->targetViews.back();
            if (byRun) {
                if (!run.depthChecked || run.recorder != recorderId || run.generation != generation || run.depthAddress != state.depth->address || run.depthView != expected || (intentOn && !depthIntentHeld(ownIntent, run.depthApplied))) refusal = AppendRefusal::Depth;
            } else {
                const auto& checked = HostThreadLocal<ExpressDepthChecked, ExpressDepthStorage>();
                if (checked.recorder != recorderId || checked.generation != generation || checked.address != state.depth->address || checked.view != expected) refusal = AppendRefusal::Depth;
                else {
                    if (intentOn && !ignoreVersions && !std::exchange(intentNoted, true)) noteDepthIntent(ownIntent, checked.applied, false);
                    if (intentOn && !depthIntentHeld(ownIntent, checked.applied)) refusal = AppendRefusal::Depth;
                    else conditionsApplied = checked.applied;
                }
            }
        }
        if (refusal == AppendRefusal::None) {
            const auto& taken = *preparedInputs;
            const std::size_t indexRanges = draw.indexed ? 1u : 0u;
            if (taken.recipe.get() != recipe || taken.device != context.device || taken.state != &state || taken.draw != &draw || taken.shaders != shaders.data() || taken.stages != shaders.size() || taken.ranges.size() < indexRanges) refusal = AppendRefusal::InputShape;
            else if ((taken.forget & 1u) != 0 || GuestMemory::ForgetSerial() != taken.forget || HostImportSetGeneration() != taken.imports || GuestAllocations::GuestAllocationsGeneration_nid_postfix() != taken.mapping) refusal = AppendRefusal::Imports;
            else if ((context.copiedWriters != nullptr && !context.copiedWriters->empty()) || !DrawCopiedWriters()->empty()) refusal = AppendRefusal::Writers;
            if (trusted && refusal == AppendRefusal::None) NoteCommitterTrusted();
            for (std::size_t at = 0; refusal == AppendRefusal::None && at < taken.ranges.size(); ++at) {
                const auto& range = taken.ranges[at];
                const auto bytes = static_cast<std::size_t>(range.end - range.begin);
                if (range.source != nullptr) {
                    // The proof calls the flush hook, which can end the pass the tests above held.
                    if (!mirroredInputHolds(context, range, at < indexRanges)) refusal = AppendRefusal::Imports;
                    else if (recorder->Id() != recorderId || recorder->RenderPassGeneration() != generation || recorder->AppendablePass(express->passKey) == VK_NULL_HANDLE) refusal = AppendRefusal::Pass;
                    continue;
                }
                const bool vertexRange = at >= indexRanges && at + taken.indirectRanges < taken.ranges.size();
                const bool rangeTrusted = vertexRange && (trusted || inputsCheck.Skip());
                const bool rangeAnswered = validation && !rangeTrusted && range.import != nullptr && inputRangeAnswered(*recorder, range.begin, bytes, WriteValidation::Test::LeaderInputs);
                if (range.import == nullptr) refusal = AppendRefusal::Imports;
                else if (!rangeTrusted && !rangeAnswered && !PendingZones::Clear(range.begin, bytes) && (recorder->PendingWriteOverlaps(range.begin, bytes) || PendingStorageOverlaps(range.begin, bytes, nullptr))) refusal = AppendRefusal::InputWrites;
                else if (!rangeTrusted && !rangeAnswered && (recorder->PendingLabelIn(range.begin, bytes) || recorder->QueuedStoreOverlaps(range.begin, bytes))) refusal = AppendRefusal::InputStores;
                if (vertexRange && (refusal == AppendRefusal::InputWrites || refusal == AppendRefusal::InputStores)) inputsCheck.Fail();
            }
        }
        if (refusal == AppendRefusal::None && trusted) {
            NoteCommitterTrusted();
            if ((prepared->forget & 1u) != 0 || GuestMemory::ForgetSerial() != prepared->forget) refusal = AppendRefusal::Buffers;
        } else if (refusal == AppendRefusal::None && !(buffersCheck.Armed() && preparedMapped() && buffersCheck.Skip()) && !(validation ? resources.PreparedZonesClearByWindows(*prepared, *recorder) : resources.PreparedZonesClear(*prepared, *recorder))) {
            refusal = AppendRefusal::Buffers;
            if (buffersCheck.Audits() && preparedMapped()) buffersCheck.Fail();
        }
        return refusal;
    };
    if (resume == nullptr && RuntimeSwitches::Set(RuntimeSwitch::AppendCommits) && !atFloor && !verifyDue && !reprove && templateMiss == ShaderResources::ExpressMiss::None) {
        conditionsRecorder = recorder->Id();
        conditionsGeneration = recorder->RenderPassGeneration();
        const auto generation = conditionsGeneration;
        auto refusal = conditions(false, false);
        if (refusal == AppendRefusal::None && !resources.PreparedInPlaceHolds(*prepared, *recorder, inPlaceBehind, &buffersCheck, inPlaceValidation)) {
            refusal = AppendRefusal::Buffers;
            inPlaceRefused = true;
        }
        followSeen = expectAppend && (refusal == AppendRefusal::Depth || refusal == AppendRefusal::Multisample) ? conditions(true, false) : refusal;
        if (expectAppend && (followKind & 2u) != 0 && followSeen == AppendRefusal::Targets) {
            auto seen = conditions(false, true);
            if (seen == AppendRefusal::Depth || seen == AppendRefusal::Multisample) seen = conditions(true, true);
            if (seen == AppendRefusal::None) followSeen = seen;
        }
        if (refusal != AppendRefusal::None) noteAppendRefusal(refusal);
        else if (appendVerifyDue()) appendClaim.armed = true;
        else appended = true;
        if (appended) noteValidated(generation);
        if (appended) depthApplied = conditionsApplied;
        for (std::size_t index = 0; appended && index < keep.targets.size(); ++index) {
            if (keep.targets[index] == nullptr) NoteMultisampleTargetDrawn_nid_no_patch(keep.multisample[index]);
        }
    }
    if (resume != nullptr && resume->upload && RuntimeSwitches::Set(RuntimeSwitch::AppendCommits) && !atFloor && !verifyDue && !reprove && templateMiss == ShaderResources::ExpressMiss::None) noteAppendRefusal(AppendRefusal::Targets);
    bool followVerifyPending = false;
    const char* followContext = expectAppend ? runFollowContext(run, express) : "";
    if (expectAppend && !verifyDue) {
        if (followSeen == AppendRefusal::None) followVerifyPending = true;
        else if ((followKind & 1u) != 0 && followSeen == AppendRefusal::Count && reprove) followVerifyPending = true;
        else {
            followVerifyOpen = false;
            const bool templateMoved = followSeen == AppendRefusal::Count && templateMiss != ShaderResources::ExpressMiss::None && resources.ExpressCurrentFollowing() != ShaderResources::ExpressMiss::None;
            if (templateMoved) {
                static std::atomic<std::uint32_t> moved{0};
                if (moved.fetch_add(1, std::memory_order_relaxed) < 16u) std::fprintf(stderr, "[run-follow-verify] the template stopped being current between the follower's test and the full commit; not counted as differing\n");
            }
            noteRunFollowVerify(templateMoved, followSeen == AppendRefusal::Count ? "its append conditions were not reached" : AppendRefusalNames[static_cast<std::size_t>(followSeen)], followContext, "", followKind);
        }
    }
    const auto verifyAfterSteps = [&] {
        if (!followVerifyOpen || sideEffectsMoved) return;
        if (templateMiss == ShaderResources::ExpressMiss::None || resources.ExpressCurrentFollowing() == ShaderResources::ExpressMiss::None) {
            conditionsRecorder = recorder->Id();
            conditionsGeneration = recorder->RenderPassGeneration();
            auto seen = conditions(false, false);
            if (seen == AppendRefusal::Depth || seen == AppendRefusal::Multisample) seen = conditions(true, false);
            if (seen == AppendRefusal::None) {
                sideEffectsBegun = false;
                return;
            }
            followVerifyOpen = false;
            followVerifyPending = false;
            noteRunFollowVerify(false, AppendRefusalNames[static_cast<std::size_t>(seen)], followContext, "the append conditions refused after the follower's steps on ", followKind);
            return;
        }
        followVerifyOpen = false;
        followVerifyPending = false;
        noteRunFollowVerify(true, "the template is not current after the proof", followContext, "", followKind);
    };
    const auto resumeHolds = [&] {
        const bool samePass = recorder->RenderPassOpen() && recorder->Id() == run.recorder && recorder->RenderPassGeneration() == run.generation;
        for (std::size_t index = 0; index < keep.targets.size(); ++index) {
            const auto& target = keep.targets[index];
            if (target == nullptr || index >= state.colors.size() || index >= express->targetVersions.size() || index >= express->dccAddresses.size() || !target->Cached() || target->UploadVersion() != express->targetVersions[index]) return false;
            const auto dcc = SurfaceForTarget(state.colors[index]).dccAddress;
            if (dcc != express->dccAddresses[index] || !StorageImageServesKeys(*target, dcc) || (samePass && !target->NotedInPass(run.recorder, run.generation))) return false;
        }
        return resume->templateMiss != ShaderResources::ExpressMiss::None || resources.ExpressCurrent() == ShaderResources::ExpressMiss::None || resources.ExpressCurrentFollowing() != ShaderResources::ExpressMiss::None;
    };
    bool depthStamped = false;
    std::uint32_t multisampleStamped = 0;
    const bool sameTargets = appended || (atFloor && recorder->RenderPassOpen() && floorPass.recorder == recorder->Id() && floorPass.generation == recorder->RenderPassGeneration() && floorPass.framebuffer == keep.framebuffer.get());
    const bool targetsAhead = resume != nullptr && resume->targets;
    if (targetsAhead) {
        out.refreshed = resume->refreshed;
        ++run.census.resumedTargets;
        if (resume->miss == DrawExpressMiss::Targets) {
            ++run.census.resumedMisses;
            return miss(DrawExpressMiss::Targets);
        }
        if (runFollowVerifyDue()) noteRunFollowVerify(resumeHolds(), "a target or the template answer moved since the follower's steps", runFollowContext(run, express), "the express commit resumed, but ", std::uint8_t{4});
    }
    if (followKind != 0) sideEffectsBegun = true;
    for (std::size_t index = 0; !sameTargets && !targetsAhead && index < keep.targets.size(); ++index) {
        const auto& target = keep.targets[index];
        if (target == nullptr) {
            if (index >= state.colors.size() || index >= keep.multisample.size() || keep.multisample[index] == nullptr || index >= express->multisampleImages.size() || index >= recipe->targetViews.size()) return targetMiss(0);
            auto& multisampleMemo = HostThreadLocal<ExpressMultisampleMemo, ExpressMultisampleStorage>();
            auto& checked = multisampleMemo.checked;
            auto& nextChecked = multisampleMemo.next;
            const auto* owner = keep.multisample[index].get();
            const auto found = std::ranges::find_if(checked, [&](const ExpressMultisampleChecked& entry) { return entry.owner == owner; });
            if (!recorder->RenderPassOpen() || found == checked.end() || found->recorder != recorder->Id() || found->generation != recorder->RenderPassGeneration()) {
                const auto generation = recorder->RenderPassGeneration();
                std::shared_ptr<void> current;
                VkImage image = VK_NULL_HANDLE;
                if (state.colors[index].fmaskAddress != 0) GuardOneFragmentFmask(context, state.colors[index]);
                const auto view = MultisampleTargetView(context, state.colors[index], current, &image);
                if (view != recipe->targetViews[index] || current.get() != owner || image != express->multisampleImages[index]) return targetMiss(1);
                auto& slot = found != checked.end() ? *found : checked[nextChecked++ % checked.size()];
                slot = {owner, recorder->Id(), generation};
                multisampleStamped |= 1u << static_cast<std::uint32_t>(&slot - checked.data());
                ++out.refreshed;
                noteValidated(generation);
            } else {
                NoteMultisampleTargetDrawn_nid_no_patch(keep.multisample[index]);
                noteValidated(recorder->RenderPassGeneration());
            }
            continue;
        }
        if (const auto cause = refreshExpressTarget(context, state, *express, target, index, *recorder, out.refreshed); cause != ExpressTargetHeld) return targetMiss(cause);
    }
    if (followKind != 0) sideEffectsMoved = !recorder->RenderPassOpen() || recorder->RenderPassGeneration() != run.generation;
    if (followKind == 2u) verifyAfterSteps();
    const bool provedAhead = resume != nullptr && resume->proved;
    if (!provedAhead) {
        if (express->hasDepth && (DepthSurfacesSerial() != express->depthSerial || DepthSurfacesReleased() != express->depthReleased)) return miss(DrawExpressMiss::Depth);
        if (!sameTargets && recipe->hasDepth && state.depth.has_value()) {
            auto& checked = HostThreadLocal<ExpressDepthChecked, ExpressDepthStorage>();
            const auto expected = recipe->targetViews.empty() ? VK_NULL_HANDLE : recipe->targetViews.back();
            const bool keyed = recorder->RenderPassOpen() && checked.recorder == recorder->Id() && checked.generation == recorder->RenderPassGeneration() && checked.address == state.depth->address && checked.view == expected;
            if (keyed && intentOn && !std::exchange(intentNoted, true)) noteDepthIntent(ownIntent, checked.applied, false);
            if (!keyed || (intentOn && !depthIntentHeld(ownIntent, checked.applied))) {
                if (keyed) depthIntentViews.fetch_add(1, std::memory_order_relaxed);
                const auto generation = recorder->RenderPassGeneration();
                const auto view = DepthSurfaceViewWithStencilWriteIntent(context, *state.depth, state.depthWrite, state.stencilTest, writesStencilTarget(state), overwrittenStencil(state));
                if (view != expected) return miss(DrawExpressMiss::Depth);
                checked = {recorder->Id(), generation, state.depth->address, expected, static_cast<std::uint8_t>(keyed ? checked.applied | ownIntent : ownIntent)};
                depthApplied = checked.applied;
                depthStamped = true;
                ++out.refreshed;
                noteValidated(generation);
            } else {
                depthApplied = checked.applied;
                noteValidated(recorder->RenderPassGeneration());
            }
        }
        if (StorageTexture::DrawnLinearPending()) return miss(DrawExpressMiss::Linear);
        if (prepared == nullptr || !RuntimeSwitches::Set(RuntimeSwitch::DirectBuffers) || record.verifyBuffers || prepared->plan == nullptr || prepared->plan != resources.DirectBufferPlan(shaders)) return miss(DrawExpressMiss::Buffers);
        if (preparedInputs == nullptr || !RuntimeSwitches::Set(RuntimeSwitch::DirectInputs) || record.verifyInputs) return miss(DrawExpressMiss::Inputs);
        if (verifyDue) {
            out.verify = true;
            out.reproved = reprove;
            return miss(DrawExpressMiss::Verify);
        }
    } else if (resume->depthChecked) {
        depthApplied = HostThreadLocal<ExpressDepthChecked, ExpressDepthStorage>().applied;
        noteValidated(resume->depthGeneration);
    }
    const DepthCopySourceScope depthCopySource(state.depthCopySource);
    PerformanceTimer timing("Graphics.ExpressCommit");
    DrawTimer timer(false);
    if (provedAhead) {
        ++run.census.resumedProofs;
        record.effects = {};
        if (resume->miss == DrawExpressMiss::Template) {
            ++run.census.resumedMisses;
            return miss(DrawExpressMiss::Template);
        }
        record.effects.proved = true;
        record.effects.cleanProof = resume->cleanProof;
        out.result = std::move(resume->result);
        out.reproved = true;
        out.reproveMiss = templateMiss;
    } else {
        recorder->BoundKeptBytes();
        record.effects = {};
        if (reprove) {
            if (!proveRecordTemplate(record, resources, shaders, out.result, false)) {
                record.effects = {};
                out.result = {};
                return miss(DrawExpressMiss::Template);
            }
            out.reproved = true;
            out.reproveMiss = templateMiss;
            if ((followKind & 1u) != 0) {
                sideEffectsMoved = sideEffectsMoved || !recorder->RenderPassOpen() || recorder->RenderPassGeneration() != run.generation;
                verifyAfterSteps();
            }
        }
    }
    DrawInputs inputs;
    const bool leanOn = RuntimeSwitches::Set(RuntimeSwitch::LeanExpress);
    const auto taken = appended ? DrawRecordInputs::Used : takePreparedInputs(context, state, draw, shaders, *recipe, *preparedInputs, inputs, timer, timing, atFloor, leanOn, &inputsCheck);
    if (appended) {
        inputs.indexBytes = preparedInputs->indexBytes;
        const bool guardsSkipped = trusted || inputsCheck.Skip();
        const bool inputsAnswered = !guardsSkipped && !audit && !validation && carriedInputGuards(*preparedInputs, FollowAheadSite::Leader);
        for (std::size_t at = 0; at < preparedInputs->ranges.size(); ++at) {
            const auto& range = preparedInputs->ranges[at];
            const auto bytes = static_cast<std::size_t>(range.end - range.begin);
            if (range.source != nullptr) {
                recorder->Keep(range.source);
                continue;
            }
            if (!guardsSkipped && !inputsAnswered && at + preparedInputs->indirectRanges < preparedInputs->ranges.size() && !(validation && inputGuardsAnswered(recorder, range.begin, bytes, WriteValidation::Test::LeaderGuards))) requireInputGuards(&inputsCheck, range.begin, bytes, at == 0 && draw.indexed ? "an index buffer read" : "a vertex buffer read");
            recorder->NotePendingRead(range.begin, bytes, Recorder::ReadKind::DrawInput);
        }
        if (trusted) NoteCommitterTrusted();
    }
    if (taken != DrawRecordInputs::Used && taken != DrawRecordInputs::Rechecked) {
        record.effects = {};
        return miss(DrawExpressMiss::Inputs);
    }
    const auto giveBack = [&](DrawExpressMiss why) {
        if (!appended) {
            preparedInputs->vertexInput = std::move(inputs.vertexInput);
            preparedInputs->vertexHandles = std::move(inputs.vertexHandles);
            preparedInputs->vertexOffsets = std::move(inputs.vertexOffsets);
            preparedInputs->inPlaceReads = std::move(inputs.inPlaceReads);
        }
        record.effects = {};
        return miss(why);
    };
    record.effects.inputs = taken;
    bool untouched = appended || (leanOn && (prepared->forget & 1u) == 0 && GuestMemory::ForgetSerial() == prepared->forget && resources.PreparedUntouched(*prepared));
    if (untouched && leanExpressVerifyDue()) {
        const bool clear = resources.PreparedMovesClear(*prepared, *recorder, nullptr) && resources.UnmovedRangesClear(*prepared, *recorder);
        noteLeanExpressVerify(clear);
        untouched = clear;
    }
    if (leanOn) noteLeanExpress(untouched);
    const bool movesTrusted = trusted && (prepared->forget & 1u) == 0 && GuestMemory::ForgetSerial() == prepared->forget;
    if (movesTrusted && !atFloor && !untouched) NoteCommitterTrusted();
    if (!atFloor && !untouched && !movesTrusted && !(buffersCheck.Armed() && preparedMapped() && buffersCheck.Skip())) {
        auto ticket = record.ticket;
        ticket.ranges = &record.ticketRanges;
        ticket.checked = 0;
        ticket.trusted = 0;
        const auto movesByWindows = validation ? resources.PreparedMovesClearByWindows(*prepared, *recorder, &ticket, WriteValidation::Test::LeaderMoves) : std::optional<bool>{};
        if (!(movesByWindows.has_value() ? *movesByWindows : resources.PreparedMovesClear(*prepared, *recorder, &ticket))) {
            if (buffersCheck.Audits() && preparedMapped()) buffersCheck.Fail();
            return giveBack(DrawExpressMiss::Moved);
        }
        ticket.ranges = nullptr;
        record.ticket = ticket;
        record.effects.movedChecked = ticket.checked;
        record.effects.movedTrusted = ticket.trusted;
    }
    if (!appended && (inPlaceRefused || !resources.PreparedInPlaceHolds(*prepared, *recorder, inPlaceBehind, &buffersCheck, inPlaceValidation))) return giveBack(DrawExpressMiss::Moved);
    const auto indirectTouches = [&](std::uint64_t begin, std::uint64_t end) {
        if (!indirectExpress) return false;
        const auto& args = *draw.indirect;
        return (begin < args.arguments + args.RangeBytes() && args.arguments < end) || (args.countIndirect && begin < args.countAddress + 4u && args.countAddress < end);
    };
    const auto touches = [&](std::uint64_t begin, std::uint64_t end) {
        const auto bytes = static_cast<std::size_t>(end - begin);
        return resources.WritesOverlap(begin, bytes) || resources.ReadsOverlap(begin, bytes) || ShaderResources::PreparedInPlaceOverlaps(*prepared, begin, end) || indirectTouches(begin, end);
    };
    const bool storesTrusted = trusted && !indirectExpress;
    if (storesTrusted) NoteCommitterTrusted();
    const bool storesSkipped = storesTrusted || (!indirectExpress && buffersCheck.Skip());
    constexpr auto storesTest = WriteValidation::Test::LeaderStores;
    const bool storesArmable = validation && !storesSkipped && !indirectExpress && !prepared->inPlace && (recorder->HasQueuedKeyStores() || recorder->HasQueuedStores());
    const bool storesHeld = storesArmable && resources.StoresArmHolds(*recorder);
    const bool storesVerify = storesHeld && WriteValidation::VerifyDue(storesTest);
    const bool storesAnswered = storesHeld && !storesVerify;
    const auto storesBefore = Recorder::StoreGenerationForValidation();
    bool storesFlushed = false;
    if (!storesSkipped && !storesAnswered && recorder->HasQueuedKeyStores() && recorder->AnyQueuedKeyStore(touches)) {
        if (!indirectExpress) buffersCheck.Fail();
        storesFlushed = true;
        recorder->FlushKeyStores();
    }
    if (!storesSkipped && !storesAnswered && recorder->HasQueuedStores() && recorder->AnyQueuedStore(touches)) {
        if (!indirectExpress) buffersCheck.Fail();
        storesFlushed = true;
        recorder->FlushStores();
    }
    if (storesVerify) {
        WriteValidation::NoteVerify(storesTest, storesFlushed, "a queued store or key store touches the template's buffers");
    } else if (storesAnswered) {
        WriteValidation::Note(storesTest, WriteValidation::Outcome::Answered);
    } else if (storesArmable && !storesFlushed) {
        resources.ArmStores(*recorder, storesBefore);
        WriteValidation::Note(storesTest, WriteValidation::Outcome::Armed);
    }
    ValidationGuards selectionGuards{*recorder, WriteValidation::Test::LeaderSelection};
    const ShaderResources::GuardTests selectionTests{&validationDepthGuards, &validationMultisampleGuards, &selectionGuards, &validationWrites};
    if ((validation ? resources.FragmentSelectionHoldsBy(*recorder, prepared->moved, *prepared, untouched, &selectionTests, &buffersCheck) : resources.FragmentSelectionHolds(*recorder, prepared->moved, *prepared, nullptr, untouched, &buffersCheck, !trusted && !audit ? carriedBufferGuards(*prepared, FollowAheadSite::Leader) : std::uint8_t{0})) != ShaderResources::PreparedBuffers::Selection::Taken) return giveBack(DrawExpressMiss::Selection);
    if (resources.ChecksFaults() && fragment->faultGeneration != resources.FaultGeneration()) resources.RewriteFragmentFaults(*fragment->bindings);
    try {
        if (!reprove) resources.ExpressStage(*recorder);
    } catch (const std::exception&) {
        return giveBack(DrawExpressMiss::Staging);
    }
    if (!atFloor && !(prepared->aliasesClear && RuntimeSwitches::Set(RuntimeSwitch::AliasesAtPrepare))) CheckBufferAliases(shaders, state.color, draw.indexAddress, inputs.indexBytes);
    if (recipe->hasDepth && writesDepthTarget(state)) NoteDepthSurfaceWrite(context, *state.depth);
    if (const auto present = Recorder::Presents(); record.touchedPresent != present) {
        if ((!atFloor && !RuntimeSwitches::Set(RuntimeSwitch::TemplateTouch)) || resources.TouchedPresent() != present) {
            SharedResourceCache().TouchObject(recipe->key, resources);
            resources.NoteTouchedPresent(present);
        }
        record.touchedPresent = present;
    }
    countCache(&DrawProfile::cacheHits);
    keepExpressBindings(*recorder, fragment->bindings);
    if (express->keptRecorder != recorder->Id() || express->keptBatch != recorder->OpenSerial()) {
        recorder->Keep(express->keep);
        for (const auto& owner : keep.multisample) {
            if (owner != nullptr) KeepMultisampleTarget_nid_no_patch(owner);
        }
        express->keptRecorder = recorder->Id();
        express->keptBatch = recorder->OpenSerial();
    }
    if (trusted) NoteCommitterTrusted();
    else if (buffersCheck.Skip()) {
    } else if (buffersCheck.Audits()) {
        resources.RequireGuestBuffersAudited(buffersCheck);
    } else resources.RequireGuestBuffers();
    const auto& inPlaceReads = appended ? preparedInputs->inPlaceReads : inputs.inPlaceReads;
    bool passSeen = appended && RuntimeSwitches::Set(RuntimeSwitch::PassReadsOnce) && express->passSeenRecorder == recorder->Id() && express->passSeenGeneration == recorder->RenderPassGeneration() && express->writes.empty() && express->writeOnly.empty() && express->images.empty() && recorder->PassPureReads() && recorder->AppendablePass(express->passKey) != VK_NULL_HANDLE;
    const bool passSeenVerify = passSeen && appendVerifyDue();
    if (passSeenVerify) passSeen = false;
    if (followVerifyPending) passSeen = false;
    auto& passReads = HostThreadLocal<std::vector<std::pair<std::uint64_t, std::uint64_t>>, ExpressReadsStorage>();
    passReads.clear();
    struct ExpressMovedReadsStorage {};
    auto& movedReads = HostThreadLocal<std::vector<std::pair<std::uint64_t, std::uint64_t>>, ExpressMovedReadsStorage>();
    movedReads.clear();
    if (prepared->inPlace) {
        for (const auto& entry : prepared->moved) {
            if (entry.inPlace != VK_NULL_HANDLE) movedReads.emplace_back(entry.begin, entry.begin + entry.bytes);
        }
    }
    if (!passSeen) {
        passReads.assign(express->reads.begin(), express->reads.end());
        passReads.insert(passReads.end(), inPlaceReads.begin(), inPlaceReads.end());
        passReads.insert(passReads.end(), movedReads.begin(), movedReads.end());
    }
    auto& passAttachments = HostThreadLocal<std::vector<std::pair<VkImage, bool>>, ExpressAttachmentsStorage>();
    passAttachments.clear();
    for (std::size_t index = 0; index < keep.targets.size(); ++index) passAttachments.emplace_back(keep.targets[index] != nullptr ? keep.targets[index]->Image() : express->multisampleImages[index], true);
    const PassAccess passAccess{passReads, express->writes, express->images, passAttachments, false, false, express->writeOnly};
    const auto generationBeforeStart = recorder->RenderPassGeneration();
    const bool passOpenBeforeStart = recorder->RenderPassOpen() && !recorder->RunOpen();
    const bool followPassHeld = followVerifyPending && recorder->RenderPassOpen() && recorder->Id() == run.recorder && generationBeforeStart == run.generation;
    Recorder::DrawPassStart start;
    if (passSeen) {
        recorder->NotePassReads(inPlaceReads);
        if (!movedReads.empty()) recorder->NotePassReads(movedReads);
        start = {recorder->AppendablePass(express->passKey), true, false, PassBreak::Continued};
    } else {
        start = recorder->StartDrawPass(express->passKey, indirectExpress && !recorder->PassIndirectReads(), express->fenced, passAccess, express->restartDepth);
        if (followVerifyPending) noteRunFollowVerify((followKind != 0 && sideEffectsMoved) || (followPassHeld && start.continued), followPassHeld ? PassBreakNames[static_cast<std::size_t>(start.reason)] : "the pass ended inside the express commit", followContext, "", followKind);
        if (passSeenVerify) {
            const auto checked = passReadsChecked.fetch_add(1, std::memory_order_relaxed) + 1u;
            const auto differing = start.continued ? passReadsDiffering.load(std::memory_order_relaxed) : passReadsDiffering.fetch_add(1, std::memory_order_relaxed) + 1u;
            if (!start.continued && differing <= 16u) std::fprintf(stderr, "[pass-reads-verify] a commit that would have skipped the pass access did not continue the pass: %s\n", PassBreakNames[static_cast<std::size_t>(start.reason)]);
            if (checked % 4096u == 0u) std::fprintf(stderr, "[pass-reads-verify] %llu checked, %llu differing\n", static_cast<unsigned long long>(checked), static_cast<unsigned long long>(differing));
        }
    }
    if ((depthStamped || multisampleStamped != 0) && passOpenBeforeStart && recorder->RenderPassGeneration() == generationBeforeStart + 1 && RuntimeSwitches::Set(RuntimeSwitch::ExpressPassStamps)) {
        const auto opened = recorder->RenderPassGeneration();
        if (depthStamped) {
            auto& depthMemo = HostThreadLocal<ExpressDepthChecked, ExpressDepthStorage>();
            if (depthMemo.recorder == recorder->Id() && depthMemo.generation == generationBeforeStart) depthMemo.generation = opened;
        }
        if (multisampleStamped != 0) {
            auto& memo = HostThreadLocal<ExpressMultisampleMemo, ExpressMultisampleStorage>();
            for (std::size_t slot = 0; slot < memo.checked.size(); ++slot) {
                auto& entry = memo.checked[slot];
                if (((multisampleStamped >> slot) & 1u) != 0 && entry.recorder == recorder->Id() && entry.generation == generationBeforeStart) entry.generation = opened;
            }
        }
    }
    record.access.generation = recorder->RenderPassGeneration();
    if (atFloor) floorPass = {recorder->Id(), recorder->RenderPassGeneration(), keep.framebuffer.get()};
    const auto commands = start.commands;
    for (const auto& [begin, end] : movedReads) recorder->NotePendingRead(begin, static_cast<std::size_t>(end - begin), Recorder::ReadKind::DrawInput);
    auto drawTiming = Recorder::NoTiming;
    if (!start.continued) {
        recorder->PrepareSampleSlot();
        PerformanceTrace::Count(PerformanceTrace::RenderPassScope);
        drawTiming = recorder->BeginGpuTiming(Recorder::CommandClass::Draw);
        if (start.barrier) {
            const bool passIndirect = RuntimeSwitches::Set(RuntimeSwitch::IndirectInPass);
            const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | (passIndirect ? static_cast<VkAccessFlags>(VK_ACCESS_INDIRECT_COMMAND_READ_BIT) : VkAccessFlags{0})};
            context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
            Recorder::CountBarriers(Recorder::CommandClass::Draw, 1);
            if (passIndirect) recorder->NotePassIndirectReads();
        } else if (inPlaceBehind) {
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT);
            Recorder::CountBarriers(Recorder::CommandClass::Draw, 1);
        }
        if (indirectExpress && !recorder->BegunPassIndirectReads()) {
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
            Recorder::CountBarriers(Recorder::CommandClass::Draw, 1);
            recorder->NotePassIndirectReads();
        }
        if (inputs.vertexInputBehindGpuWrite) {
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT);
            Recorder::CountBarriers(Recorder::CommandClass::Draw, 1);
        }
        const bool optimalColors = RuntimeSwitches::Set(RuntimeSwitch::XpAttachmentLayouts) && keep.pipeline->DynamicRendering() && recorder->BeginPassLayouts(commands, passAttachments);
        const bool optimalDepth = beginDepthLayout(context, *recorder, commands, *keep.pipeline, *keep.framebuffer, state, resources);
        keep.pipeline->BeginPass(commands, *keep.framebuffer, state.renderExtent, state, optimalColors, optimalDepth);
    }
    NoteHeapRecord(false, false);
    if (context.recorder != nullptr) context.recorder->NoteSampledDraw(commands);
    if (auto* stream = CommandStream::FromHandle(commands); stream != nullptr) {
        stream->Splice(std::move(fragment->commands));
        record.effects.fragment.spliced = true;
    } else {
        CommandStream::ReplayFragment(fragment->commands, commands);
        record.effects.fragment.replayed = true;
    }
    recorder->GraphicsState().Invalidate();
    record.effects.fragment.offered = true;
    NoteGuestWorkIn(context, commands);
    if (!passSeen) recorder->LeaveRenderPassOpen(express->passKey, drawTiming, express->fenced, passAccess);
    express->passSeenRecorder = recorder->Id();
    express->passSeenGeneration = recorder->RenderPassGeneration();
    record.effects.keptOnce = true;
    const auto recorderId = recorder->Id();
    for (std::size_t index = 0; index < keep.targets.size(); ++index) {
        auto* target = keep.targets[index].get();
        if (target == nullptr) continue;
        const auto [slice, count] = express->slices[index];
        if (target->RenoteInPass(recorderId, recorder->RenderPassGeneration(), slice, count)) {
            ++record.effects.notesSkipped;
            continue;
        }
        target->MarkSliceDirty(slice, count);
        if (recorder->RenderPassOpen()) target->RememberPassNote(recorderId, recorder->RenderPassGeneration(), slice, count);
    }
    record.effects.proved = true;
    if (!reprove) record.effects.cleanProof = ShaderResources::CleanProof::Skipped;
    if (resources.ChecksFaults() && resources.TakeFaultCheck(recorderId, recorder->OpenSerial())) recorder->OnComplete([kept = keep.resources] { kept->CheckFaults(); });
    record.effects.hold = DrawRecordHold::Held;
    record.effects.passHeld = true;
    record.effects.buffers = DrawRecordBuffers::Used;
    record.effects.pushed = fragment->pushing;
    if (recorder->RenderPassOpen() && GuestMemory::CollectEpoch() != 0) {
        auto& remembered = record.pass;
        remembered.valid = true;
        remembered.recorder = recorderId;
        remembered.generation = recorder->RenderPassGeneration();
        remembered.epoch = GuestMemory::CollectEpoch();
        remembered.proof = GuestMemory::ProofSerial();
        remembered.pending = StorageTexture::PendingSerial();
        remembered.depthSerial = DepthSurfacesSerial();
        remembered.depthReleased = DepthSurfacesReleased();
        remembered.versions.clear();
        for (const auto& target : keep.targets) remembered.versions.push_back(target != nullptr ? target->UploadVersion() : 0u);
        if (std::ranges::any_of(keep.multisample, [](const std::shared_ptr<void>& owner) { return owner != nullptr; })) remembered.valid = false;
    } else {
        record.pass.valid = false;
    }
    timing.Mark("draw_and_resource_release");
    if (RuntimeSwitches::Set(RuntimeSwitch::AppendCommits)) noteAppendCommit(appended);
    appendClaim.confirmed = out.refreshed == 0 && taken == DrawRecordInputs::Used && !reprove;
    if (indirectExpress) {
        CountIndirectDraw(IndirectDrawPath::Gpu, 0.0, false);
        if (start.continued) countIndirectInPass();
    }
    ShaderResources::NotePreparedInPlaceTaken(*prepared);
    if (run.open != 0 && !atFloor && RuntimeSwitches::Set(RuntimeSwitch::RunFollow)) {
        const auto opened = recorder->RenderPassGeneration();
        const bool needs = (recipe->hasDepth && state.depth.has_value()) || std::ranges::any_of(keep.targets, [](const std::shared_ptr<StorageTexture>& target) { return target == nullptr; });
        const bool held = !needs || (validated && !validatedSplit && (validatedAt == opened || (validatedAt == generationBeforeStart && passOpenBeforeStart && opened == generationBeforeStart + 1)));
        static_cast<void>(leadExpressRun(run, *recorder, *express, keep, state, *recipe, held, depthApplied));
    }
    if (following && !expectAppend && !appendClaim.armed && !passSeenVerify) {
        ++(appended ? run.census.appended : run.census.full);
        if (leadTimed) {
            const auto nanoseconds = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - leadBegan).count());
            (appended ? run.census.appendedNs : run.census.fullNs) += nanoseconds;
            ++(appended ? run.census.appendedTimed : run.census.fullTimed);
        }
    }
    out.taken = true;
    out.result.recorded = true;
    return out;
}

void BeginExpressRun() {
    auto& run = expressRun();
    if (run.open++ != 0 || !run.dropped) return;
    run.dropped = false;
    const bool items = std::exchange(run.itemsBetween, false);
    const auto* recorder = Recorder::Active();
    const bool following = RuntimeSwitches::Set(RuntimeSwitch::RunFollow);
    if (run.led && following && !RuntimeSwitches::Set(RuntimeSwitch::RunFollowNoCarry) && GuestMemory::GpuMutex().HeldByThisThread() && (items || GuestMemory::GpuHoldSerial() == run.dropSerial + 1) && recorder != nullptr && recorder->Id() == run.recorder && recorder->OpenSerial() == run.batch && recorder->RenderPassGeneration() == run.generation && recorder->RenderPassOpen()) {
        ++run.census.carried;
        if (items) ++run.census.carriedItems;
        run.overItems = items;
        return;
    }
    if (run.led) ++run.census.notCarried;
    finishRun(run, following ? run.dropReason : ExpressRunEnd::Switch);
}

void EndExpressRun(ExpressRunEnd reason) {
    auto& run = expressRun();
    if (run.open == 0 || --run.open != 0) return;
    run.segmentOpen = false;
    run.expectAppend = false;
    run.expectKind = 0;
    run.resume.record = nullptr;
    run.intentNoted = false;
    if (!run.led) return;
    const bool following = RuntimeSwitches::Set(RuntimeSwitch::RunFollow);
    if (!following || reason == ExpressRunEnd::Stop || reason == ExpressRunEnd::Switch || RuntimeSwitches::Set(RuntimeSwitch::RunFollowNoCarry) || !GuestMemory::GpuMutex().HeldByThisThread()) {
        finishRun(run, following ? reason : ExpressRunEnd::Switch);
        return;
    }
    run.dropped = true;
    run.itemsBetween = false;
    run.dropReason = reason;
    run.dropSerial = GuestMemory::GpuHoldSerial();
}

void BreakExpressRun() {
    finishRun(expressRun(), ExpressRunEnd::Refusal);
}

bool ExpressRunLed() {
    const auto& run = expressRun();
    return run.led && !run.dropped;
}

void RefuseExpressFollow(ExpressRunRefusal why) {
    auto& run = expressRun();
    if (run.led && !run.dropped) refuseRun(run, why);
}

std::uint64_t ExpressFollowAttempts() {
    return expressRun().census.seen;
}

void NoteExpressFollowItem(std::uint64_t nanoseconds) {
    auto& census = expressRun().census;
    census.itemNs += nanoseconds;
    ++census.itemTimed;
}

void NoteExpressRunHold(std::uint64_t nanoseconds) {
    auto& census = expressRun().census;
    census.holdNs += nanoseconds;
    ++census.holds;
}

void NoteExpressRunItem() {
    auto& run = expressRun();
    if (run.dropped) run.itemsBetween = true;
}

void NoteExpressNonFollower(bool attempted, std::uint64_t attemptNanoseconds, std::uint64_t commitNanoseconds) {
    auto& run = expressRun();
    auto& census = run.census;
    if (!attempted) {
        census.unledNs += commitNanoseconds;
        ++census.unledTimed;
        return;
    }
    if (run.refused == ExpressRunRefusal::Verify) return;
    census.refusingNs += attemptNanoseconds;
    ++census.refusingTimed;
    census.fallbackNs += commitNanoseconds;
    ++census.fallbackTimed;
}

void DropExpressFollowVerify() {
    auto& run = expressRun();
    run.expectAppend = false;
    run.expectKind = 0;
    run.resume.record = nullptr;
    run.intentNoted = false;
    run.refused = ExpressRunRefusal::Count;
}

void NoteExpressFollowDue(bool due) {
    expressRun().followDue = due;
}

namespace {

std::atomic<std::uint64_t> committerTrusted{0};

}

void NoteCommitterTrusted(std::uint64_t count) {
    committerTrusted.store(committerTrusted.load(std::memory_order_relaxed) + count, std::memory_order_relaxed);
}

std::uint64_t TakeCommitterTrusted() {
    return committerTrusted.exchange(0, std::memory_order_relaxed);
}

bool ExpressFollowCommit(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, DrawRecordCommit& record, const ShaderResources::PreparedBuffers& prepared, PreparedInputs& preparedInputs, DrawFragment& fragment) {
    if (const auto hook = expressHook.load(std::memory_order_acquire); hook != nullptr) hook(record);
    auto& run = expressRun();
    if (!run.led || run.dropped) return false;
    if (!RuntimeSwitches::Set(RuntimeSwitch::RunFollow)) {
        finishRun(run, ExpressRunEnd::Switch);
        return false;
    }
    const bool timed = (++run.followOrdinal & 7u) == 0;
    const bool staged = (run.followOrdinal & 63u) == 36u;
    const auto began = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::array<std::chrono::steady_clock::time_point, 9> marks{};
    const auto mark = [&marks, staged](std::size_t at) {
        if (staged) marks[at] = std::chrono::steady_clock::now();
    };
    mark(0);
    const auto refuse = [&run, timed, began](ExpressRunRefusal why) {
        if (timed && why != ExpressRunRefusal::Verify) {
            run.census.refusedNs += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - began).count());
            ++run.census.refusedTimed;
        }
        refuseRun(run, why);
        return false;
    };
    run.resume.record = nullptr;
    auto* express = record.express.get();
    const auto* recipe = record.recipe.get();
    if (express == nullptr || !express->valid || express->keep == nullptr || recipe == nullptr || express->recipe != recipe || express->state != &state || record.expressRefused || record.verifyBuffers || record.verifyInputs) return refuse(ExpressRunRefusal::State);
    if (draw.indirect) return refuse(ExpressRunRefusal::Indirect);
    if (prepared.inPlace) return refuse(ExpressRunRefusal::InPlace);
    auto* recorder = Recorder::Active();
    if (recorder == nullptr || context.recorder != recorder || express->device != context.device) return refuse(ExpressRunRefusal::Device);
    auto& keep = *express->keep;
    const auto recorderId = recorder->Id();
    const bool windows = RuntimeSwitches::Set(RuntimeSwitch::FollowWindows) && !VerifyGuardSpans();
    const bool widerCensus = RuntimeSwitches::Set(RuntimeSwitch::FollowWider);
    const bool trusted = RuntimeSwitches::Set(RuntimeSwitch::XpTrustCommitter);
    const bool audit = !trusted && keep.resources != nullptr && RuntimeSwitches::Set(RuntimeSwitch::TrustAudit);
    const bool validation = WriteValidation::Session() != 0;
    auto templateCheck = audit ? keep.resources->TrustCheck(CommitterTrustFamily::Template, CommitterTrustSite::FollowerTemplate) : CommitterTrustCheck{};
    auto buffersCheck = audit ? keep.resources->TrustCheck(CommitterTrustFamily::Buffers, CommitterTrustSite::FollowerBuffers) : CommitterTrustCheck{};
    auto inputsCheck = audit ? keep.resources->TrustCheck(CommitterTrustFamily::Inputs, CommitterTrustSite::FollowerInputs) : CommitterTrustCheck{};
    const auto preparedMapped = [&] { return (prepared.forget & 1u) == 0 && GuestMemory::ForgetSerial() == prepared.forget; };
    const bool intentOn = RuntimeSwitches::Set(RuntimeSwitch::ExpressDepthIntent);
    const std::uint8_t ownIntent = intentOn ? depthIntent(state) : 0u;
    const auto depthMemoHolds = [&] {
        if (!recipe->hasDepth || !state.depth.has_value()) return true;
        const auto& checked = HostThreadLocal<ExpressDepthChecked, ExpressDepthStorage>();
        const auto expected = recipe->targetViews.empty() ? VK_NULL_HANDLE : recipe->targetViews.back();
        return recorder->RenderPassOpen() && checked.recorder == recorder->Id() && checked.generation == recorder->RenderPassGeneration() && checked.address == state.depth->address && checked.view == expected && (!intentOn || depthIntentHeld(ownIntent, checked.applied));
    };
    int widened = -1;
    const auto widen = [&] {
        if (widened < 0) widened = widerCensus && followWiderAllowed(run, keep, draw, fragment) && depthMemoHolds() ? 1 : 0;
        return widened == 1;
    };
    std::uint64_t generation = 0;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    bool upload = false;
    auto templateMiss = ShaderResources::ExpressMiss::None;
    const auto holds = [&](bool first) {
        const auto ownTargets = [&](std::size_t cause) {
            if (first && widerCensus) {
                ++run.census.ownTargets[cause];
                if (std::ranges::any_of(keep.multisample, [](const std::shared_ptr<void>& owner) { return owner != nullptr; })) ++run.census.ownTargets[3];
            }
            return ExpressRunRefusal::OwnTargets;
        };
        if (express->switches != run.switches || RuntimeSwitches::Generation() != run.switches || recordChecksQueued() || DescriptorHeapEnabled()) return ExpressRunRefusal::Switches;
        if (fragment.bindings == nullptr || fragment.device != context.device || fragment.resources.get() != keep.resources.get() || fragment.pipeline.get() != keep.pipeline.get()) return ExpressRunRefusal::Objects;
        if (recorderId != run.recorder) return ExpressRunRefusal::Recorder;
        generation = recorder->RenderPassGeneration();
        if (generation != run.generation) return ExpressRunRefusal::Generation;
        if (express->passKey != run.passKey) return ExpressRunRefusal::Key;
        if (!express->writes.empty() || !express->writeOnly.empty()) return ExpressRunRefusal::OwnWrites;
        if (std::ranges::any_of(express->images, [](const std::pair<VkImage, bool>& image) { return image.second; })) return ExpressRunRefusal::OwnImages;
        commands = recorder->AppendablePass(express->passKey);
        if (commands == VK_NULL_HANDLE) return ExpressRunRefusal::NotAppendable;
        const auto forget = GuestMemory::ForgetSerial();
        const auto imports = HostImportSetGeneration();
        const auto mapping = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
        const auto depthSerial = DepthSurfacesSerial();
        const auto depthReleased = DepthSurfacesReleased();
        if (forget != run.forget || imports != run.imports || mapping != run.mapping || depthSerial != run.depthSerial || depthReleased != run.depthReleased) return ExpressRunRefusal::Serials;
        if (express->hasDepth && (express->depthSerial != depthSerial || express->depthReleased != depthReleased)) return ExpressRunRefusal::OwnSerials;
        if (StorageTexture::DrawnLinearPending()) return ExpressRunRefusal::Linear;
        if ((context.copiedWriters != nullptr && !context.copiedWriters->empty()) || !DrawCopiedWriters()->empty()) return ExpressRunRefusal::Writers;
        const auto targetCount = keep.targets.size();
        if (targetCount != run.targets || express->targetVersions.size() < targetCount || express->dccAddresses.size() < targetCount) return ownTargets(0);
        for (std::size_t index = 0; index < targetCount; ++index) {
            const auto* target = keep.targets[index].get();
            if (target == nullptr) {
                if (index >= keep.multisample.size() || keep.multisample[index] == nullptr || static_cast<const void*>(keep.multisample[index].get()) != run.owners[index] || index >= express->multisampleImages.size() || express->multisampleImages[index] != run.images[index] || index >= state.colors.size() || index >= recipe->targetViews.size()) return ExpressRunRefusal::Multisample;
                continue;
            }
            if (static_cast<const void*>(target) != run.owners[index] || index >= state.colors.size()) return ownTargets(0);
            if (target->Image() != run.images[index] || target->UploadVersion() != run.versions[index] || !target->Cached() || !target->NotedInPass(recorderId, generation)) return ExpressRunRefusal::Targets;
            if (express->dccAddresses[index] != run.dcc[index]) return ownTargets(2);
            if (express->targetVersions[index] != run.versions[index]) {
                if (!first || !RuntimeSwitches::Set(RuntimeSwitch::ExpressTargetUploads) || !widen()) return ownTargets(1);
                upload = true;
            }
            if (!StorageImageServesKeys(*target, express->dccAddresses[index])) return ExpressRunRefusal::Targets;
        }
        if (recipe->hasDepth && state.depth.has_value()) {
            const auto expected = recipe->targetViews.empty() ? VK_NULL_HANDLE : recipe->targetViews.back();
            if (!run.depthChecked || state.depth->address != run.depthAddress || expected != run.depthView) return ExpressRunRefusal::Depth;
            if (intentOn && !depthIntentHeld(ownIntent, run.depthApplied)) {
                if (first) {
                    noteDepthIntent(ownIntent, run.depthApplied, true);
                    run.intentNoted = true;
                }
                return ExpressRunRefusal::Depth;
            }
        }
        if (first) mark(1);
        auto& resources = *keep.resources;
        if (validation && first) WriteValidation::Note(WriteValidation::Test::TemplateCurrent, WriteValidation::Outcome::Kept);
        if (const auto missed = trusted ? resources.ExpressCurrentTrusted() : templateCheck.Armed() ? resources.ExpressCurrentAudited(templateCheck, true) : resources.ExpressCurrentFollowing(); missed != ShaderResources::ExpressMiss::None) {
            const bool reprovable = missed != ShaderResources::ExpressMiss::Regions && (missed != ShaderResources::ExpressMiss::NotProved || (RuntimeSwitches::Set(RuntimeSwitch::ExpressKeepState) && resources.ExpressWatchable()));
            if (first && widerCensus) {
                ++run.census.templateMisses[static_cast<std::size_t>(missed)];
                if (reprovable) ++run.census.templateReprovable;
            }
            if (!first || !reprovable || !widen()) return ExpressRunRefusal::Template;
            templateMiss = missed;
        }
        if (first) mark(2);
        if (!RuntimeSwitches::Set(RuntimeSwitch::DirectBuffers) || !RuntimeSwitches::Set(RuntimeSwitch::DirectInputs) || prepared.plan == nullptr || prepared.plan != resources.DirectBufferPlan(shaders) || !preparedInputsUsable(context, preparedInputs)) return ExpressRunRefusal::Prepared;
        const std::size_t indexRanges = draw.indexed ? 1u : 0u;
        if (preparedInputs.recipe.get() != recipe || preparedInputs.device != context.device || preparedInputs.state != &state || preparedInputs.draw != &draw || preparedInputs.shaders != shaders.data() || preparedInputs.stages != shaders.size() || preparedInputs.ranges.size() < indexRanges || (preparedInputs.forget & 1u) != 0 || forget != preparedInputs.forget || imports != preparedInputs.imports || mapping != preparedInputs.mapping) return ExpressRunRefusal::Inputs;
        if (trusted) NoteCommitterTrusted(2);
        const bool passSkipped = trusted || buffersCheck.Skip();
        constexpr auto passTest = WriteValidation::Test::FollowerPass;
        const bool passChecked = validation && !passSkipped;
        const auto passSession = passChecked ? WriteValidation::Session() : std::uint64_t{0};
        const auto passSwitches = RuntimeSwitches::Generation();
        const auto passBatch = recorder->OpenSerial();
        const auto passSerial = recorder->PassHazardSerial();
        const bool passHeld = passChecked && passSession != 0 && express->wvPassSession == passSession && express->wvPassSwitches == passSwitches && express->wvPassRecorder == recorderId && express->wvPassBatch == passBatch && express->wvPassGeneration == generation && express->wvPassSerial == passSerial && express->wvPassAttachments == run.wvAttachments;
        const bool passVerify = passHeld && WriteValidation::VerifyDue(passTest);
        const bool readsHeld = passHeld && !passVerify && express->wvPassNoWrites;
        const bool imagesHeld = passHeld && !passVerify;
        if (!passSkipped && !readsHeld && !recorder->PassReadsFollow(express->reads, preparedInputs.inPlaceReads)) {
            if (passVerify && express->wvPassNoWrites) WriteValidation::NoteVerify(passTest, true, "a write of the pass overlaps the draw's reads");
            buffersCheck.Fail();
            return ExpressRunRefusal::PassWrites;
        }
        if (!passSkipped && !imagesHeld && !recorder->PassImagesFollow(express->images, std::span<const std::pair<VkImage, bool>>(run.attachments.data(), run.targets))) {
            if (passVerify) WriteValidation::NoteVerify(passTest, true, "an image of the draw or of the run conflicts with the images or attachments of the pass");
            buffersCheck.Fail();
            return ExpressRunRefusal::ImageHazard;
        }
        if (passVerify) WriteValidation::NoteVerify(passTest, false, nullptr);
        if (passChecked && passSession != 0) {
            if (passHeld) {
                WriteValidation::Note(passTest, WriteValidation::Outcome::Answered);
            } else {
                express->wvPassSession = passSession;
                express->wvPassSwitches = passSwitches;
                express->wvPassRecorder = recorderId;
                express->wvPassBatch = passBatch;
                express->wvPassGeneration = generation;
                express->wvPassSerial = passSerial;
                express->wvPassAttachments = run.wvAttachments;
                express->wvPassNoWrites = recorder->PassNoWrites();
                WriteValidation::Note(passTest, WriteValidation::Outcome::Armed);
            }
        }
        if (express->fenced && !recorder->PassFenced()) return ExpressRunRefusal::Fenced;
        keyRunRanges(run.clear, recorderId, recorder->WriteNoteCount(), StorageTexture::PendingSerial(), Recorder::QueuedLabelSerial());
        if (windows) keyRunWindows(run, *recorder);
        const bool queuedStores = recorder->HasQueuedStores();
        if (trusted) NoteCommitterTrusted();
        for (const auto& range : preparedInputs.ranges) {
            const auto bytes = static_cast<std::size_t>(range.end - range.begin);
            if (range.source != nullptr) {
                if (!mirroredInputHolds(context, range, draw.indexed && &range == &preparedInputs.ranges.front())) return ExpressRunRefusal::Inputs;
                continue;
            }
            if (range.import == nullptr) return ExpressRunRefusal::Inputs;
            const bool vertexRange = !draw.indexed || &range != &preparedInputs.ranges.front();
            if (vertexRange && (trusted || inputsCheck.Skip())) continue;
            const auto hazard = [&](ExpressRunRefusal why) {
                if (vertexRange) inputsCheck.Fail();
                return why;
            };
            if (validation && inputRangeAnswered(*recorder, range.begin, bytes, WriteValidation::Test::FollowerInputs)) continue;
            if (queuedStores && recorder->QueuedStoreOverlaps(range.begin, bytes)) return hazard(ExpressRunRefusal::InputStores);
            if (runRangesHold(run.clear, range.begin, range.end, true)) {
                ++run.census.rangesSkipped;
                continue;
            }
            const auto pendingOver = [&] { return !PendingZones::Clear(range.begin, bytes) && (recorder->PendingWriteOverlaps(range.begin, bytes) || PendingStorageOverlaps(range.begin, bytes, nullptr)); };
            bool pendingClear = windows && runWindowsClear(run, *recorder, WindowFact::Pending, range.begin, bytes);
            if (pendingClear && windowVerifyDue()) {
                pendingClear = !pendingOver();
                noteWindowVerify(WindowFact::Pending, pendingClear);
            }
            if (!pendingClear && pendingOver()) return hazard(ExpressRunRefusal::InputWrites);
            bool labelsClear = windows && runWindowsClear(run, *recorder, WindowFact::Labels, range.begin, bytes);
            if (labelsClear && windowVerifyDue()) {
                labelsClear = !recorder->PendingLabelIn(range.begin, bytes);
                noteWindowVerify(WindowFact::Labels, labelsClear);
            }
            if (!labelsClear && recorder->PendingLabelIn(range.begin, bytes)) return hazard(ExpressRunRefusal::InputStores);
            addRunRange(run.clear, range.begin, range.end);
        }
        if (first) mark(3);
        FollowZones zones{run, *recorder, windows, validation};
        if (trusted) {
            NoteCommitterTrusted(2);
            if ((prepared.forget & 1u) != 0 || GuestMemory::ForgetSerial() != prepared.forget) return ExpressRunRefusal::Buffers;
        } else if (!(buffersCheck.Armed() && preparedMapped() && buffersCheck.Skip()) && !resources.PreparedZonesClearBy(prepared, &followZoneClear, &zones)) {
            if (buffersCheck.Audits() && preparedMapped()) buffersCheck.Fail();
            return ExpressRunRefusal::Buffers;
        }
        if (!trusted && !buffersCheck.Skip() && (recorder->HasQueuedKeyStores() || queuedStores)) {
            constexpr auto storesTest = WriteValidation::Test::FollowerStores;
            const bool storesHeld = validation && resources.StoresArmHolds(*recorder);
            const bool storesVerify = storesHeld && WriteValidation::VerifyDue(storesTest);
            const auto storesBefore = Recorder::StoreGenerationForValidation();
            if (storesHeld && !storesVerify) {
                WriteValidation::Note(storesTest, WriteValidation::Outcome::Answered);
            } else {
                const auto touches = [&](std::uint64_t begin, std::uint64_t end) {
                    const auto bytes = static_cast<std::size_t>(end - begin);
                    return resources.WritesOverlap(begin, bytes) || resources.ReadsOverlap(begin, bytes);
                };
                if ((recorder->HasQueuedKeyStores() && recorder->AnyQueuedKeyStore(touches)) || (queuedStores && recorder->AnyQueuedStore(touches))) {
                    if (storesVerify) WriteValidation::NoteVerify(storesTest, true, "a queued store or key store touches the template's buffers");
                    buffersCheck.Fail();
                    return ExpressRunRefusal::Stores;
                }
                if (storesVerify) {
                    WriteValidation::NoteVerify(storesTest, false, nullptr);
                } else if (validation) {
                    resources.ArmStores(*recorder, storesBefore);
                    WriteValidation::Note(storesTest, WriteValidation::Outcome::Armed);
                }
            }
        }
        return ExpressRunRefusal::Count;
    };
    if (const auto why = holds(true); why != ExpressRunRefusal::Count) return refuse(why);
    auto& resources = *keep.resources;
    const bool reproving = templateMiss != ShaderResources::ExpressMiss::None;
    const auto kind = static_cast<std::uint8_t>((reproving ? 1u : 0u) | (upload ? 2u : 0u));
    if (runFollowVerifyDue()) {
        run.expectAppend = true;
        run.expectKind = kind;
        return refuse(ExpressRunRefusal::Verify);
    }
    mark(4);
    auto& resume = run.resume;
    const auto late = [&](ExpressRunRefusal why) {
        if (kind != 0 && (resume.targets || resume.proved)) {
            resume.record = &record;
            ++run.census.widerRefused;
        }
        return refuse(why);
    };
    if (kind != 0) {
        resume.ordinal = run.followOrdinal;
        resume.targets = true;
        resume.proved = false;
        resume.upload = upload;
        resume.depthChecked = false;
        resume.depthGeneration = 0;
        resume.miss = DrawExpressMiss::None;
        resume.refreshed = 0;
        resume.templateMiss = templateMiss;
        resume.cleanProof.reset();
        resume.result = {};
        for (std::size_t index = 0; index < keep.targets.size(); ++index) {
            if (const auto cause = refreshExpressTarget(context, state, *express, keep.targets[index], index, *recorder, resume.refreshed); cause != ExpressTargetHeld) {
                expressTargetCauses[cause].fetch_add(1, std::memory_order_relaxed);
                resume.miss = DrawExpressMiss::Targets;
                return late(ExpressRunRefusal::OwnTargets);
            }
        }
    }
    if (reproving) {
        if (express->hasDepth && (DepthSurfacesSerial() != express->depthSerial || DepthSurfacesReleased() != express->depthReleased)) return late(ExpressRunRefusal::OwnSerials);
        if (!depthMemoHolds()) return late(ExpressRunRefusal::Depth);
        resume.depthChecked = recipe->hasDepth && state.depth.has_value();
        resume.depthGeneration = recorder->RenderPassGeneration();
        if (StorageTexture::DrawnLinearPending()) return late(ExpressRunRefusal::Linear);
        if (!RuntimeSwitches::Set(RuntimeSwitch::DirectBuffers) || record.verifyBuffers || prepared.plan == nullptr || prepared.plan != resources.DirectBufferPlan(shaders)) return late(ExpressRunRefusal::Prepared);
        if (!RuntimeSwitches::Set(RuntimeSwitch::DirectInputs) || record.verifyInputs) return late(ExpressRunRefusal::Inputs);
    } else if (kind != 0) {
        if (const auto why = holds(false); why != ExpressRunRefusal::Count) return late(why);
    }
    const DepthCopySourceScope depthCopySource(state.depthCopySource);
    recorder->BoundKeptBytes();
    if (reproving) {
        record.effects = {};
        resume.proved = true;
        if (!proveRecordTemplate(record, resources, shaders, resume.result, false)) {
            record.effects = {};
            resume.result = {};
            resume.miss = DrawExpressMiss::Template;
            return late(ExpressRunRefusal::Template);
        }
        resume.cleanProof = record.effects.cleanProof;
        if (const auto why = holds(false); why != ExpressRunRefusal::Count) return late(why);
    }
    FollowGuards guards{run, *recorder, validation};
    const ShaderResources::GuardTests guardTests{&followDepthGuards, &followMultisampleGuards, &guards};
    ValidationGuards selectionGuards{*recorder, WriteValidation::Test::FollowerSelection};
    const ShaderResources::GuardTests selectionTests{&validationDepthGuards, &validationMultisampleGuards, &selectionGuards, &validationWrites};
    if (windows) keyRunGuardWindows(run, *recorder);
    const bool plainGuards = !trusted && !audit && !windows && !validation;
    const auto selection = windows ? resources.FragmentSelectionHoldsBy(*recorder, prepared.moved, prepared, true, &guardTests, &buffersCheck) : validation ? resources.FragmentSelectionHoldsBy(*recorder, prepared.moved, prepared, true, &selectionTests, &buffersCheck) : resources.FragmentSelectionHolds(*recorder, prepared.moved, prepared, nullptr, true, &buffersCheck, plainGuards ? carriedBufferGuards(prepared, FollowAheadSite::Follower) : std::uint8_t{0});
    if (selection != ShaderResources::PreparedBuffers::Selection::Taken) return late(ExpressRunRefusal::Selection);
    mark(5);
    try {
        if (!reproving) resources.ExpressStage(*recorder);
    } catch (const std::exception&) {
        return late(ExpressRunRefusal::Staging);
    }
    if (windows) keyRunGuardWindows(run, *recorder);
    if (trusted) NoteCommitterTrusted();
    const bool guardsSkipped = trusted || inputsCheck.Skip();
    const bool inputsAnswered = !guardsSkipped && plainGuards && carriedInputGuards(preparedInputs, FollowAheadSite::Follower);
    for (std::size_t at = 0; !guardsSkipped && !inputsAnswered && at < preparedInputs.ranges.size(); ++at) {
        const auto& range = preparedInputs.ranges[at];
        const auto bytes = static_cast<std::size_t>(range.end - range.begin);
        if (range.source != nullptr) continue;
        if (validation && inputGuardsAnswered(recorder, range.begin, bytes, WriteValidation::Test::FollowerGuards)) continue;
        if (windows && runWindowsClear(run, *recorder, WindowFact::DepthReads, range.begin, bytes) && runWindowsClear(run, *recorder, WindowFact::MultisampleSpans, range.begin, bytes)) {
            if (!windowVerifyDue()) continue;
            const bool depthIdle = DepthReadGuardIdle_nid_no_patch(range.begin, bytes);
            const bool multisampleIdle = !MayOverlapMultisampleGuards(range.begin, bytes);
            noteWindowVerify(WindowFact::DepthReads, depthIdle);
            noteWindowVerify(WindowFact::MultisampleSpans, multisampleIdle);
            if (depthIdle && multisampleIdle) continue;
        }
        requireInputGuards(&inputsCheck, range.begin, bytes, at == 0 && draw.indexed ? "an index buffer read" : "a vertex buffer read");
    }
    if (resources.ChecksFaults() && fragment.faultGeneration != resources.FaultGeneration()) resources.RewriteFragmentFaults(*fragment.bindings);
    if (!(prepared.aliasesClear && RuntimeSwitches::Set(RuntimeSwitch::AliasesAtPrepare))) CheckBufferAliases(shaders, state.color, draw.indexAddress, preparedInputs.indexBytes);
    if (recipe->hasDepth && writesDepthTarget(state)) NoteDepthSurfaceWrite(context, *state.depth);
    if (trusted) NoteCommitterTrusted();
    else if (buffersCheck.Skip()) {
    } else if (buffersCheck.Audits()) {
        resources.RequireGuestBuffersAudited(buffersCheck);
    } else resources.RequireGuestBuffers();
    if (recorder->AppendablePass(express->passKey) != commands || recorder->RenderPassGeneration() != generation) return late(ExpressRunRefusal::PassMoved);
    mark(6);
    const auto batch = recorder->OpenSerial();
    keyRunRanges(run.noted, recorderId, batch, recorder->OpenBatchGeneration(), 0);
    for (const auto& range : preparedInputs.ranges) {
        if (range.source != nullptr) {
            recorder->Keep(range.source);
            continue;
        }
        if (runRangesHold(run.noted, range.begin, range.end, false)) {
            ++run.census.readsSkipped;
            continue;
        }
        recorder->NotePendingRead(range.begin, static_cast<std::size_t>(range.end - range.begin), Recorder::ReadKind::DrawInput);
        addRunRange(run.noted, range.begin, range.end);
    }
    for (std::size_t index = 0; index < keep.targets.size(); ++index) {
        if (keep.targets[index] == nullptr) NoteMultisampleTargetDrawn_nid_no_patch(keep.multisample[index]);
    }
    if (const auto present = Recorder::Presents(); record.touchedPresent != present) {
        if (!RuntimeSwitches::Set(RuntimeSwitch::TemplateTouch) || resources.TouchedPresent() != present) {
            SharedResourceCache().TouchObject(recipe->key, resources);
            resources.NoteTouchedPresent(present);
        }
        record.touchedPresent = present;
    }
    countCache(&DrawProfile::cacheHits);
    keepExpressBindings(*recorder, fragment.bindings);
    if (express->keptRecorder != recorderId || express->keptBatch != batch) {
        recorder->Keep(express->keep);
        for (const auto& owner : keep.multisample) {
            if (owner != nullptr) KeepMultisampleTarget_nid_no_patch(owner);
        }
        express->keptRecorder = recorderId;
        express->keptBatch = batch;
    }
    if (express->passSeenRecorder != recorderId || express->passSeenGeneration != generation) {
        recorder->NotePassReads(express->reads);
        if (!express->images.empty()) recorder->NotePassImages(express->images);
    }
    keyRunRanges(run.passReads, recorderId, generation, 0, 0);
    if (std::ranges::any_of(preparedInputs.inPlaceReads, [&](const std::pair<std::uint64_t, std::uint64_t>& read) { return !runRangesHold(run.passReads, read.first, read.second, true); })) {
        recorder->NotePassReads(preparedInputs.inPlaceReads);
        for (const auto& [begin, end] : preparedInputs.inPlaceReads) {
            if (!runRangesHold(run.passReads, begin, end, true)) addRunRange(run.passReads, begin, end);
        }
    }
    record.access.generation = generation;
    mark(7);
    NoteHeapRecord(false, false);
    recorder->NoteSampledDraw(commands);
    if (auto* stream = CommandStream::FromHandle(commands); stream != nullptr) {
        stream->Splice(std::move(fragment.commands));
    } else {
        CommandStream::ReplayFragment(fragment.commands, commands);
    }
    recorder->GraphicsState().Invalidate();
    NoteGuestWorkIn(context, commands);
    express->passSeenRecorder = recorderId;
    express->passSeenGeneration = generation;
    std::size_t renoted = 0;
    for (std::size_t index = 0; index < keep.targets.size(); ++index) {
        auto* target = keep.targets[index].get();
        if (target == nullptr) continue;
        const auto [slice, count] = express->slices[index];
        if (target->RenoteInPass(recorderId, generation, slice, count)) {
            ++renoted;
            continue;
        }
        target->MarkSliceDirty(slice, count);
        if (recorder->RenderPassOpen()) target->RememberPassNote(recorderId, generation, slice, count);
    }
    if (FollowAheadCensus() && RuntimeSwitches::Set(RuntimeSwitch::FollowAhead)) noteFollowSegment(run, *express, keep, prepared, *recorder, renoted);
    if (resources.ChecksFaults() && resources.TakeFaultCheck(recorderId, batch)) recorder->OnComplete([kept = keep.resources] { kept->CheckFaults(); });
    record.pass.valid = false;
    if (RuntimeSwitches::Set(RuntimeSwitch::AppendCommits)) noteAppendCommit(true);
    ++run.length;
    if (!express->images.empty()) ++run.census.imageFollowers;
    if (reproving) ++run.census.widerReproved;
    if (upload) ++run.census.widerUploads;
    if (intentOn && recipe->hasDepth && state.depth.has_value()) noteDepthIntent(ownIntent, run.depthApplied, true);
    if (staged) {
        mark(8);
        for (std::size_t stage = 0; stage < run.census.stageNs.size(); ++stage) run.census.stageNs[stage] += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(marks[stage + 1] - marks[stage]).count());
        ++run.census.staged;
    }
    if (timed) {
        run.census.followNs += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - began).count());
        ++run.census.followTimed;
    }
    noteRunFollow(run, true);
    return true;
}

namespace {

std::shared_ptr<StorageTexture> metadataPassResident(const Context& context, const ColorTarget& color) {
    if (ColorTileModeIsLinear(color.tileMode) || context.detiler == nullptr) return nullptr;
    const auto resident = CachedStorageSurface(context, SurfaceForTarget(color));
    return resident != nullptr && resident->GuestBytes() == color.bytes ? resident : nullptr;
}

}

std::vector<std::pair<std::uint64_t, std::uint64_t>> PipelinedColorMetadataPassWrites(const ColorMetadataPass& pass) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
    if (pass.mode == ColorMetadataPass::Mode::Resolve) {
        if (!pass.source.has_value() || pass.targets.size() != 1) return writes;
        const auto& destination = pass.targets.front();
        if (destination.samples != 1u || destination.bytes == 0 || destination.cmaskAddress != 0) return writes;
        writes.emplace_back(destination.MemoryAddress(), destination.MemoryAddress() + destination.bytes);
        if (destination.dccAddress != 0) writes.emplace_back(destination.dccAddress, destination.dccAddress + std::max<std::size_t>(colorKeyCount(destination, destination.bytes), 1u));
        return writes;
    }
    if (pass.mode != ColorMetadataPass::Mode::EliminateFastClear && pass.mode != ColorMetadataPass::Mode::DccDecompress) return writes;
    for (const auto& color : pass.targets) {
        if (color.samples != 1u || color.bytes == 0) return {};
        writes.emplace_back(color.MemoryAddress(), color.MemoryAddress() + color.bytes);
        if (color.cmaskAddress != 0) writes.emplace_back(color.cmaskAddress, color.cmaskAddress + std::max<std::size_t>(color.cmaskBytes, 1u));
        if (color.dccAddress != 0) writes.emplace_back(color.dccAddress, color.dccAddress + std::max<std::size_t>(colorKeyCount(color, color.bytes), 1u));
    }
    return writes;
}

void RunColorMetadataPass(const Context& context, const ColorMetadataPass& pass) {
    if (pass.mode == ColorMetadataPass::Mode::Resolve) {
        Require(pass.source.has_value() && pass.targets.size() == 1, "CB resolve without its source and destination");
        const auto& destination = pass.targets.front();
        Require(!ColorTileModeIsLinear(destination.tileMode) && context.detiler != nullptr, "CB resolve into a target without a resident image is not implemented");
        const auto resident = CachedStorageSurface(context, SurfaceForTarget(destination));
        Require(resident->Attachable(), "storage format cannot be a color attachment");
        ResolveMultisampleTarget(context, *pass.source, *resident, resident->AttachmentView(destination.format, destination.mip));
        return;
    }
    if (pass.mode == ColorMetadataPass::Mode::FmaskDecompress) {
        for (const auto& color : pass.targets) MaterializeMultisampleTargetClear(context, color);
        return;
    }
    for (const auto& color : pass.targets) {
        if (color.cmaskAddress != 0) {
            const auto resident = metadataPassResident(context, color);
            if (materializeCmaskClear(context, color, resident) && resident != nullptr) resident->MarkDirty();
        }
        if (color.dccAddress == 0) continue;
        auto keys = CurrentDccKeys(color.dccAddress, color.bytes);
        if (keys == DccKeys::Uncompressed) continue;
        if (keys == DccKeys::ClearSingle) {
            Require(pass.mode == ColorMetadataPass::Mode::DccDecompress, "CB fast-clear eliminate over comp-to-single DCC keys (whether it expands them is not modeled)");
            storeSingleTexels(context, color);
            if (const auto resident = metadataPassResident(context, color)) resident->Refresh();
            continue;
        }
        Require(IsDccClear(keys), std::string("CB metadata pass over DCC keys that are ") + DccKeysName(keys) + " (per-block metadata is not modeled)");
        const auto texel = clearTexel(color, keys);
        const auto resident = metadataPassResident(context, color);
        if (resident != nullptr) {
            const char* refusal = nullptr;
            const bool current = keys == DccKeys::ClearRegister ? clearToTexel(*resident, texel, color.elementBytes, refusal, true) : StorageTexture::FindPending(color.address, color.bytes) == resident || resident->UploadedKeys() == keys;
            if (current) {
                resident->MarkDirty();
                MarkDccUncompressed(context, color.dccAddress, color.bytes, colorKeyCount(color, color.bytes));
                continue;
            }
        }
        storeClearTexels(context, color, texel);
    }
}

AddressDrawReuseCounts AddressDrawReuseCounters() {
    AddressDrawReuseCounts counts{};
    const auto& counters = addressDrawReuses();
    for (std::size_t index = 0; index < counts.size(); ++index) counts[index] = counters[index].load(std::memory_order_relaxed);
    return counts;
}

const char* AddressDrawReuseName(AddressDrawReuse outcome) {
    return static_cast<std::size_t>(outcome) < AddressDrawReuseNames.size() ? AddressDrawReuseNames[static_cast<std::size_t>(outcome)] : "?";
}

void SetSpaceKeepFailuresForTest(std::uint32_t count) {
    spaceKeepFailures.store(count, std::memory_order_relaxed);
}

void SetDrawStopAfterResolveForTest(DrawStopForTest stop) {
    stopAfterResolve.store(static_cast<int>(stop), std::memory_order_relaxed);
}

std::uint64_t TargetMemoEvents(TargetMemoEvent event) {
    return targetMemoEvents()[static_cast<std::size_t>(event)].Sum();
}

const char* TargetMemoEventName(TargetMemoEvent event) {
    return static_cast<std::size_t>(event) < TargetMemoEventNames.size() ? TargetMemoEventNames[static_cast<std::size_t>(event)] : "?";
}

std::uint64_t TargetMemoBuildsForTest() {
    return targetMemoBuilds;
}

std::uint64_t TargetMemoHitNanosecondsForTest(const Context& context, const State& state, std::size_t repeats, std::size_t& hits) {
    std::vector<TargetBinding> targets;
    std::vector<VkImageView> views;
    const auto proof = currentTargetProof(context);
    hits = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        targets.clear();
        targets.resize(state.colors.size());
        views.clear();
        if (repeatColors(context, state, proof, targets, views)) ++hits;
    }
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

std::uint64_t AddressDrawReuseVerifications() {
    return addressDrawReuseVerifications().load(std::memory_order_relaxed);
}

VertexPlanScratchState ThreadVertexPlanScratch() {
    const auto& scratch = VertexPlanLease::Resting();
    VertexPlanScratchState state;
    state.busy = VertexPlanLease::Busy();
    state.resting = scratch.fetches.size() + scratch.fetchValid.size() + scratch.fetchOf.size() + scratch.zeroed.size() + scratch.isSolo.size() + scratch.plannedFetches.size() + scratch.plannedOf.size() + scratch.rangeHandles.size() + scratch.rangeOffsets.size();
    state.capacity = scratch.fetches.capacity() + scratch.fetchValid.capacity() + scratch.fetchOf.capacity() + scratch.zeroed.capacity() + scratch.isSolo.capacity() + scratch.plannedFetches.capacity() + scratch.plannedOf.capacity() + scratch.rangeHandles.capacity() + scratch.rangeOffsets.capacity();
    return state;
}

bool DrawPipelineEnabled() {
    static const bool enabled = std::getenv("APS5_NO_DRAW_PIPELINE") == nullptr;
    return enabled;
}

bool CommitTrims() {
    return DrawPipelineEnabled();
}

FragmentOutputLookups ThreadFragmentOutputLookups() {
    return ThreadScratch<LastFragmentOutputs, LastFragmentOutputsTag>().lookups;
}


VkDrawCounts TakeVkDrawCounts() {
    VkDrawCounts counts;
    counts.direct = vkDrawCounts[static_cast<std::size_t>(VkDrawKind::Direct)].exchange(0, std::memory_order_relaxed);
    counts.indexed = vkDrawCounts[static_cast<std::size_t>(VkDrawKind::Indexed)].exchange(0, std::memory_order_relaxed);
    counts.indirect = vkDrawCounts[static_cast<std::size_t>(VkDrawKind::Indirect)].exchange(0, std::memory_order_relaxed);
    counts.mesh = vkDrawCounts[static_cast<std::size_t>(VkDrawKind::Mesh)].exchange(0, std::memory_order_relaxed);
    return counts;
}

}
