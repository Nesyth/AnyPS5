#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BindlessHeap.hpp"
#include "prx/libSceAgcDriver/Graphics/include/HeapReplay.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Sampler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "Recompiler.hpp"
#include "RuntimeAbi.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

namespace Graphics {

struct ShaderResourcesTestAccess {
    static std::vector<std::uint32_t>& Block(ShaderResources& resources, ShaderRecompiler::ShaderStage stage) { return resources.heapStages[HeapStageIndex(stage)].data.words; }
    static BindlessHeap::SpanMark DataMark(const ShaderResources& resources, ShaderRecompiler::ShaderStage stage) { return resources.heapStages[HeapStageIndex(stage)].data.mark; }
    static const std::vector<std::uint32_t>& Srt(const ShaderResources& resources, ShaderRecompiler::ShaderStage stage) { return resources.heapStages[HeapStageIndex(stage)].flat.words; }
    static BindlessHeap::SpanMark SrtMark(const ShaderResources& resources, ShaderRecompiler::ShaderStage stage) { return resources.heapStages[HeapStageIndex(stage)].flat.mark; }
    static std::uint32_t SrtOffset(const ShaderResources& resources, ShaderRecompiler::ShaderStage stage) { return resources.heapSrtOffsets[HeapStageIndex(stage)]; }
    static std::uint32_t SrtDwords(const ShaderResources& resources, ShaderRecompiler::ShaderStage stage) { return resources.heapSrtDwords[HeapStageIndex(stage)]; }
    static std::uint32_t DataOffset(const ShaderResources& resources, ShaderRecompiler::ShaderStage stage) { return resources.heapOffsets[HeapStageIndex(stage) * 2u]; }
    static std::uint32_t RunOffset(const ShaderResources& resources, ShaderRecompiler::ShaderStage stage) { return resources.heapOffsets[HeapStageIndex(stage) * 2u + 1u]; }
    static std::uint32_t RunFirst(const ShaderResources& resources, ShaderRecompiler::ShaderStage stage) { return resources.heapStages[HeapStageIndex(stage)].descriptors.first; }
    static BindlessHeap::SpanMark RunMark(const ShaderResources& resources, ShaderRecompiler::ShaderStage stage) { return resources.heapStages[HeapStageIndex(stage)].descriptors.mark; }
    static std::vector<VkDeviceAddressRangeEXT> RunRanges(const ShaderResources& resources, ShaderRecompiler::ShaderStage stage) { return resources.heapStages[HeapStageIndex(stage)].descriptors.ranges; }
    static std::vector<VkDeviceAddressRangeEXT> RangesBuiltAnew(const ShaderResources& resources, ShaderRecompiler::ShaderStage stage) {
        const auto index = HeapStageIndex(stage);
        std::vector<VkDeviceAddressRangeEXT> ranges{resources.srtRange(index)};
        resources.writeBufferAddresses({}, static_cast<std::uint32_t>(index), nullptr, &ranges);
        return ranges;
    }
    static std::optional<std::size_t> GuestAllocation(const ShaderResources& resources, std::uint64_t address) {
        for (std::size_t index = 0; index < resources.allocations.size(); ++index) {
            if (resources.allocations[index].guest && !resources.allocations[index].written && resources.allocations[index].address == address) return index;
        }
        return std::nullopt;
    }
    static const HeapSlotOwner* SlotOwner(const ShaderResources& resources, ShaderRecompiler::ShaderStage stage) {
        const VkShaderStageFlags stageBit = VulkanStage(stage);
        for (const auto& binding : resources.bindings) {
            if (!resources.ownsHeapBinding(binding, stageBit)) continue;
            for (const auto index : binding.imageAllocations) {
                if (binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER && index < resources.samplers.size() && resources.samplers[index] != nullptr) return resources.samplers[index].get();
                if (binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE && index < resources.textures.size() && resources.textures[index] != nullptr) return resources.textures[index].get();
            }
        }
        return nullptr;
    }
    static bool Masked(const ShaderResources& resources, ShaderRecompiler::ShaderStage stage, std::size_t dword) {
        if (resources.heapReplay == nullptr) return true;
        const auto& mask = resources.heapReplay->stages[HeapStageIndex(stage)].mask;
        return dword / 64u >= mask.size() || ((mask[dword / 64u] >> (dword % 64u)) & 1u) != 0u;
    }
};

}

}

namespace {

using AgcDriver::Graphics::CompiledShader;
using AgcDriver::Graphics::ReplayEvent;
using AgcDriver::Graphics::ReplayEvents;
using AgcDriver::Graphics::ReplayField;
using AgcDriver::Graphics::Require;
using AgcDriver::Graphics::ShaderResources;
using Access = AgcDriver::Graphics::ShaderResourcesTestAccess;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t Textures = 4;
constexpr std::uint32_t Stride = 32;
constexpr std::size_t SurfaceBytes = 256;
constexpr std::uint32_t Format8888UNorm = 56;
constexpr std::uint32_t Type2D = 9;
constexpr std::size_t SrtOffset = 4096;
constexpr std::size_t InputOffset = 8192;
constexpr std::size_t OutputOffset = 12288;
constexpr std::size_t HeapOffset = 65536;
constexpr std::size_t BlockBytes = (HeapOffset + std::size_t{Textures} * Stride + 65535u) & ~std::size_t{65535};
constexpr std::size_t SurfaceBlockBytes = (std::size_t{Textures} * SurfaceBytes + 65535u) & ~std::size_t{65535};

constexpr std::array<std::uint32_t, 25> Code{
    0xf4080100u, 0xfa000000u, 0xf4080200u, 0xfa000010u, 0xf4080300u, 0xfa000020u, 0xf4080700u, 0xfa000030u,
    0x34080082u, 0x7e200500u, 0x8f108510u, 0xf42c0502u, 0x20000000u, 0xbf8cc07fu,
    0xe0301000u, 0x80030504u, 0xf09c0f08u, 0x00450000u, 0xbf8c3f70u, 0x4a000105u,
    0xe0701000u, 0x80070004u, 0xbf810000u, 0u, 0u};

class GuestBlock {
public:
    GuestBlock(std::size_t bytes, bool gpu) {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, 65536));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
        constexpr std::uintptr_t alignment = 65536;
        void* mapped = mmap(nullptr, bytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped != MAP_FAILED) {
            const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
            const auto aligned = (begin + alignment - 1) & ~(alignment - 1);
            if (aligned != begin) munmap(mapped, aligned - begin);
            if (aligned + bytes != begin + bytes + alignment) munmap(reinterpret_cast<void*>(aligned + bytes), begin + alignment - aligned);
            block = reinterpret_cast<std::uint8_t*>(aligned);
            GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
        }
#endif
        Require(block != nullptr, "heap stage replay: cannot allocate a guest block");
        std::memset(block, 0, bytes);
        GuestAllocations::Mutation().Add(block, bytes, true, true, gpu);
        AgcDriver::GuestMemory::CollectWritesUncached(reinterpret_cast<std::uintptr_t>(block), bytes);
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint64_t At(std::size_t offset) const { return reinterpret_cast<std::uint64_t>(block + offset); }
    std::uint8_t* Bytes(std::size_t offset) const { return block + offset; }
    std::uint32_t* Words(std::size_t offset) const { return reinterpret_cast<std::uint32_t*>(block + offset); }

private:
    std::uint8_t* block = nullptr;
};

std::array<std::uint32_t, 4> BufferDescriptor(std::uint64_t address, std::uint32_t stride, std::uint32_t records) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), records, stride == 0u ? 0x31016facu : 0x01016facu};
}

void Prepare(const GuestBlock& block, const GuestBlock& surfaces) {
    std::memcpy(block.Words(0), Code.data(), sizeof(Code));
    for (std::uint32_t texture = 0; texture < Textures; ++texture) {
        const auto offset = static_cast<std::size_t>(texture) * SurfaceBytes;
        std::memset(surfaces.Bytes(offset), static_cast<int>(texture & 0xffu), SurfaceBytes);
        const auto surface = surfaces.At(offset);
        const std::array<std::uint32_t, 8> words{static_cast<std::uint32_t>(surface >> 8u), static_cast<std::uint32_t>((surface >> 40u) & 0xffu) | (Format8888UNorm << 20u) | (3u << 30u), 1u | (7u << 14u), 0xfacu | (Type2D << 28u), 0u, 0u, 0u, 0u};
        std::memcpy(block.Words(HeapOffset + static_cast<std::size_t>(texture) * Stride), words.data(), sizeof(words));
    }
    auto* srt = block.Words(SrtOffset);
    const auto heap = BufferDescriptor(block.At(HeapOffset), Stride, Textures);
    const auto input = BufferDescriptor(block.At(InputOffset), 0u, Lanes * 4u);
    const auto output = BufferDescriptor(block.At(OutputOffset), 0u, Lanes * 4u);
    std::copy(heap.begin(), heap.end(), srt);
    std::copy(input.begin(), input.end(), srt + 8);
    std::copy(output.begin(), output.end(), srt + 12);
    AgcDriver::GuestMemory::CollectWritesUncached(block.At(0), BlockBytes);
    AgcDriver::GuestMemory::CollectWritesUncached(surfaces.At(0), SurfaceBlockBytes);
}

ShaderRecompiler::RecompileResult Compile(const AgcDriver::VulkanDevice& device, const GuestBlock& block) {
    const auto srt = block.At(SrtOffset);
    const std::array<std::uint32_t, 2> userData{static_cast<std::uint32_t>(srt), static_cast<std::uint32_t>(srt >> 32u)};
    const std::span<const std::uint32_t> code(block.Words(0), Code.size());
    const std::array<ShaderRecompiler::MemoryRegion, 3> memory{{
        {block.At(0), std::as_bytes(code)},
        {srt, std::as_bytes(std::span<const std::uint32_t>(block.Words(SrtOffset), 64))},
        {block.At(HeapOffset), std::as_bytes(std::span<const std::uint32_t>(block.Words(HeapOffset), static_cast<std::size_t>(Textures) * Stride / 4u))},
    }};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, block.At(0), code, 0, {}},
        {Lanes, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.ComputeTarget(Lanes),
        {0, 0, 0, 128}
    };
    return ShaderRecompiler::Recompile(request);
}

std::optional<std::size_t> BindingOf(const ShaderRecompiler::RecompileResult& program, ShaderRecompiler::DescriptorRole role) {
    for (std::size_t index = 0; index < program.bindings.size(); ++index) {
        if (program.bindings[index].role == role) return index;
    }
    return std::nullopt;
}

std::uint64_t Count(ReplayEvent event) {
    return ReplayEvents(event);
}

std::uint64_t Mismatches() {
    std::uint64_t total = 0;
    for (std::size_t field = 0; field < static_cast<std::size_t>(ReplayField::Count); ++field) total += AgcDriver::Graphics::ReplayMismatches(static_cast<ReplayField>(field));
    return total;
}

void RequireMore(std::uint64_t before, ReplayEvent event, const std::string& what) {
    Require(Count(event) > before, "heap stage replay: " + what);
}

void RequireSame(std::uint64_t before, ReplayEvent event, const std::string& what) {
    Require(Count(event) == before, "heap stage replay: " + what);
}

AgcDriver::Graphics::BindlessHeap* hookHeap = nullptr;
AgcDriver::Graphics::Recorder* hookRecorder = nullptr;
int hookSubmits = 0;

void SubmitInsideTheFirstLoop() {
    if (hookSubmits == 0) return;
    --hookSubmits;
    hookHeap->Submitted(*hookRecorder, 0u);
}

struct Replay {
    AgcDriver::Graphics::Context context;
    CompiledShader built;
    std::shared_ptr<ShaderResources> resources;
    std::shared_ptr<ShaderResources> twin;
    AgcDriver::Graphics::BindlessHeap& heap;
    AgcDriver::Graphics::Recorder& recorder;
    const ShaderRecompiler::RecompileResult& program;

    CompiledShader Shader(const ShaderRecompiler::HeapStageWords& words) const { return {ShaderStage::Compute, &program, 0, nullptr, &words}; }

    void Check(const std::string& what, bool overlaid = false) const {
        const auto stage = ShaderStage::Compute;
        const auto fail = [&](const char* why) { return "heap stage replay: " + what + ": " + why; };
        auto& block = Access::Block(*resources, stage);
        Require(block == Access::Block(*twin, stage), fail("the stage data differs from the one today's refresh builds"));
        const auto mark = Access::DataMark(*resources, stage);
        Require(Access::DataOffset(*resources, stage) == mark.offset, fail("the root's shader data offset is not the stage data span"));
        Require(heap.SpanCurrent(mark), fail("the stage data span is not current"));
        Require(std::ranges::equal(heap.RingWords(mark), block), fail("the ring does not hold the stage data"));
        const auto& srt = Access::Srt(*resources, stage);
        Require(srt == Access::Srt(*twin, stage), fail("the srt words differ from the ones today's refresh writes"));
        if (!srt.empty()) {
            const auto srtMark = Access::SrtMark(*resources, stage);
            Require(Access::SrtOffset(*resources, stage) == srtMark.offset && Access::SrtDwords(*resources, stage) == srt.size(), fail("the root's srt offset or dwords are not the srt span's"));
            Require(heap.SpanCurrent(srtMark) && std::ranges::equal(heap.RingWords(srtMark), srt), fail("the ring does not hold the srt words in a current span"));
        }
        if (overlaid) return;
        Require(Access::RunOffset(*resources, stage) == Access::RunFirst(*resources, stage), fail("the root's descriptor index is not the template run's"));
        Require(heap.SpanCurrent(Access::RunMark(*resources, stage)), fail("the template run is not current"));
        const auto ranges = Access::RunRanges(*resources, stage);
        const auto expected = Access::RangesBuiltAnew(*resources, stage);
        const bool same = std::ranges::equal(ranges, expected, [](const VkDeviceAddressRangeEXT& a, const VkDeviceAddressRangeEXT& b) { return a.address == b.address && a.size == b.size; });
        Require(same, fail("the template run's ranges differ from the ranges built anew"));
    }

    void Fresh(const ShaderRecompiler::HeapStageWords& words, const std::string& what) const {
        auto fresh = std::make_shared<ShaderResources>(context, built);
        const auto shader = Shader(words);
        fresh->RefreshHeapStage(std::span<const CompiledShader>(&shader, 1));
        auto& block = Access::Block(*resources, ShaderStage::Compute);
        Require(block == Access::Block(*fresh, ShaderStage::Compute), "heap stage replay: " + what + ": the stage data differs from a build from scratch");
        Require(std::ranges::equal(heap.RingWords(Access::DataMark(*resources, ShaderStage::Compute)), Access::Block(*fresh, ShaderStage::Compute)), "heap stage replay: " + what + ": the ring words differ from a build from scratch");
    }

    void Draw(const ShaderRecompiler::HeapStageWords& words, const std::string& what, bool submitBeforePass = false) {
        const auto shader = Shader(words);
        const std::span<const CompiledShader> shaders(&shader, 1);
        const auto proof = resources->ReplayHeapStages(shaders, false);
        if (submitBeforePass) heap.Submitted(recorder, 0u);
        resources->ConfirmHeapReplay(recorder, shaders, nullptr, proof);
        twin->RefreshHeapStage(shaders);
        Check(what);
    }
};

int Run(const AgcDriver::VulkanDevice& device, const GuestBlock& block, std::string_view mode) {
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    if (!context.descriptorHeap || !context.heapBufferDescriptors) {
        std::puts("skipped, the device keeps no buffer descriptors in the shader data ring, so no stage takes the heap stage replay");
        return VulkanTestSkipped;
    }
    Require(context.recorder != nullptr, "heap stage replay: the device has no recorder");
    const auto program = Compile(device, block);
    Require(program.gpuResolvedImages && program.dataAddressPush, "heap stage replay: the program is not in the heap's GPU-resolved form");
    const auto dataBinding = BindingOf(program, ShaderRecompiler::DescriptorRole::ShaderData);
    Require(dataBinding.has_value() && !program.bindings[*dataBinding].guestDescriptor.empty(), "heap stage replay: the program has no shader data");
    const auto& dataWords = program.bindings[*dataBinding].guestDescriptor;
    const auto level = AgcDriver::Graphics::ReplayVerifyLevel();
    const bool strict = AgcDriver::Graphics::ReplayVerifyStrict();
    if (mode != "replay") Require(level == (mode == "shadow" ? 1 : 2), "heap stage replay: the verify mode does not have the level the test runs with");
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    const CompiledShader built{ShaderStage::Compute, &program, 0};
    Replay replay{context, built, std::make_shared<ShaderResources>(context, built), std::make_shared<ShaderResources>(context, built), AgcDriver::Graphics::BindlessHeap::For(context), *context.recorder, program};
    Require(replay.resources->HeapStage(), "heap stage replay: the set was not built on the heap");
    hookHeap = &replay.heap;
    hookRecorder = &replay.recorder;
    AgcDriver::Graphics::SetHeapReplayRefreshHookForTest(&SubmitInsideTheFirstLoop);
    const auto userDwords = std::min<std::size_t>(dataWords.size(), 8u);
    ShaderRecompiler::HeapStageWords first;
    first.userData.assign(dataWords.begin(), dataWords.begin() + static_cast<std::ptrdiff_t>(userDwords));

    const auto created = Count(ReplayEvent::NewRecord);
    replay.Draw(first, "the first replay draw");
    RequireMore(created, ReplayEvent::NewRecord, "the first draw did not create the template's record");

    const auto stages = Count(ReplayEvent::Stages);
    const auto proven = Count(ReplayEvent::DataProven);
    replay.Draw(first, "a repeated draw");
    RequireMore(stages, ReplayEvent::Stages, "a repeated draw did not replay its stage");
    RequireMore(proven, ReplayEvent::DataProven, "a repeated draw in the same batch did not keep its stage data span by the submit epoch");

    std::vector<std::size_t> free;
    std::optional<std::size_t> maskedSlotWord;
    for (std::size_t dword = 0; dword < userDwords; ++dword) {
        if (!Access::Masked(*replay.resources, ShaderStage::Compute, dword)) free.push_back(dword);
    }
    for (const auto& word : program.heapSlotWords) {
        if (word.dword < userDwords && Access::Masked(*replay.resources, ShaderStage::Compute, word.dword)) maskedSlotWord = word.dword;
    }
    std::optional<std::size_t> tail;
    for (std::size_t dword = userDwords; dword < dataWords.size() && !tail.has_value(); ++dword) {
        if (!Access::Masked(*replay.resources, ShaderStage::Compute, dword)) tail = dword;
    }
    std::printf("heap stage replay: %zu shader data dwords, %zu of user data, %zu of them outside the template's patched dwords, %s, %s\n", dataWords.size(), userDwords, free.size(),
                maskedSlotWord.has_value() ? "a slot word under the user data" : "no slot word under the user data", tail.has_value() ? "a word after the user data outside the patched dwords" : "no plain word after the user data");
    Require(!free.empty(), "heap stage replay: every user data dword of the test program is patched by the template, so the patch path cannot be exercised");

    auto changed = first;
    const auto patchedCount = std::min<std::size_t>(free.size(), 3u);
    for (std::size_t index = 0; index < patchedCount; ++index) changed.userData[free[index]] ^= 0x5a5a0000u + static_cast<std::uint32_t>(index);
    const auto patched = Count(ReplayEvent::DataPatched);
    const auto rebuiltBeforePatch = Count(ReplayEvent::ShapeChanged) + Count(ReplayEvent::MaskedChange) + Count(ReplayEvent::SlotsMoved);
    replay.Draw(changed, "a draw with changed user data dwords");
    RequireMore(patched, ReplayEvent::DataPatched, "changed user data dwords were not patched into the stage data");
    Require(Count(ReplayEvent::ShapeChanged) + Count(ReplayEvent::MaskedChange) + Count(ReplayEvent::SlotsMoved) == rebuiltBeforePatch, "heap stage replay: a draw with changed unpatched user data dwords was rebuilt instead of patched");
    replay.Fresh(changed, "a draw with changed user data dwords");

    replay.heap.Submitted(replay.recorder, 0u);
    const auto held = Count(ReplayEvent::DataHeld);
    replay.Draw(changed, "a repeated draw after a submit");
    RequireMore(held, ReplayEvent::DataHeld, "a repeated draw after a submit did not hold its stage data span again");

    if (maskedSlotWord.has_value()) {
        auto masked = changed;
        masked.userData[*maskedSlotWord] ^= 0x00010000u;
        const auto maskedBefore = Count(ReplayEvent::MaskedChange);
        replay.Draw(masked, "a draw whose changed user data dword is a slot word the template patches");
        RequireMore(maskedBefore, ReplayEvent::MaskedChange, "a changed slot word under the user data did not send the stage to today's refresh");
        replay.Draw(changed, "a draw after the masked change");
    } else {
        std::puts("heap stage replay: the test program has no slot word under its user data; the masked-dword rebuild is not exercised");
    }

    const auto writes = Count(ReplayEvent::NonReplayWrites);
    auto other = first;
    other.userData[free.front()] ^= 0x00a50000u;
    {
        const auto shader = replay.Shader(other);
        replay.resources->RefreshHeapStage(std::span<const CompiledShader>(&shader, 1));
    }
    replay.Draw(other, "the first replay draw after another writer refreshed the stage");
    RequireMore(writes, ReplayEvent::NonReplayWrites, "a refresh outside the replay did not invalidate the template's record");
    replay.Draw(other, "a replay draw after the record was rebuilt");

    if (const auto srtBinding = BindingOf(program, ShaderRecompiler::DescriptorRole::FlattenedSrt); srtBinding.has_value() && !program.bindings[*srtBinding].guestDescriptor.empty()) {
        auto srtChanged = other;
        srtChanged.flattenedSrt = program.bindings[*srtBinding].guestDescriptor;
        srtChanged.flattenedSrt.back() ^= 0x00000100u;
        const auto changes = Count(ReplayEvent::SrtChanged);
        replay.Draw(srtChanged, "a draw whose flattened srt changed");
        RequireMore(changes, ReplayEvent::SrtChanged, "a changed flattened srt was not written to a new span");
        const auto same = Count(ReplayEvent::SrtSame);
        replay.Draw(srtChanged, "a repeated draw with the changed srt");
        RequireMore(same, ReplayEvent::SrtSame, "a repeated srt was not kept");
        replay.Draw(other, "a draw back on the program's srt");
    } else {
        std::puts("heap stage replay: the test program binds no flattened srt; the srt change is not exercised");
    }

    if (const auto* owner = Access::SlotOwner(*replay.resources, ShaderStage::Compute); owner != nullptr) {
        const auto moved = Count(ReplayEvent::SlotsMoved);
        replay.heap.ReleaseSlots(owner);
        replay.Draw(other, "a draw after the heap released a slot the stage binds");
        RequireMore(moved, ReplayEvent::SlotsMoved, "a released slot the stage binds did not send the stage to today's refresh");
        replay.Draw(other, "a replay draw after the slots were taken again");
    } else {
        std::puts("heap stage replay: the stage binds no slot of its own; the slot release is not exercised");
    }

    Require(tail.has_value(), "heap stage replay: the test program's shader data has no plain word after the user data, so the per-draw words cannot differ from the program's where the block keeps them");
    auto perDraw = first;
    perDraw.bindings.assign(program.bindings.size(), {});
    perDraw.bindings[*dataBinding] = dataWords;
    perDraw.bindings[*dataBinding][*tail] ^= 0x00100000u;
    replay.Draw(first, "a program words draw before the per-draw words");
    const auto reshaped = Count(ReplayEvent::ShapeChanged);
    replay.Draw(perDraw, "a draw whose per-draw shader data words differ from the program's");
    RequireMore(reshaped, ReplayEvent::ShapeChanged, "per-draw shader data words that differ from the program's did not send the stage to today's refresh");
    replay.Draw(perDraw, "a repeated draw with the per-draw words");
    const auto back = Count(ReplayEvent::ShapeChanged);
    replay.Draw(first, "a draw back on the program's shader data words");
    RequireMore(back, ReplayEvent::ShapeChanged, "program words after per-draw words did not send the stage to today's refresh");

    replay.Draw(first, "a program words draw before a submit inside the first loop");
    const auto duringRefresh = Count(ReplayEvent::SubmitDuringRefresh);
    hookSubmits = 1;
    replay.Draw(perDraw, "a per-draw words draw whose batch was submitted inside the replay's first loop");
    Require(hookSubmits == 0, "heap stage replay: the replay's first loop did not run the submit hook");
    RequireMore(duringRefresh, ReplayEvent::SubmitDuringRefresh, "a submit inside the first loop did not send the draw to today's refresh");
    replay.Draw(first, "a program words draw after the per-draw words absorbed by the submit fallback");
    auto patchedFirst = first;
    patchedFirst.userData[free.front()] ^= 0x00330000u;
    replay.Draw(patchedFirst, "a patched program words draw after the submit fallback");
    replay.Fresh(patchedFirst, "a patched program words draw after the submit fallback");

    if (const auto input = Access::GuestAllocation(*replay.resources, block.At(InputOffset)); input.has_value() && Access::GuestAllocation(*replay.twin, block.At(InputOffset)) == input) {
        const std::array<ShaderResources::MovedBuffer, 1> moved{{{*input, block.At(InputOffset + 1024u), Lanes * 4u, {}}}};
        const auto shader = replay.Shader(other);
        const std::span<const CompiledShader> shaders(&shader, 1);
        const auto overlays = Count(ReplayEvent::RunsSkipped);
        const auto missing = Count(ReplayEvent::OverlayMissing);
        const auto run = Access::RunOffset(*replay.resources, ShaderStage::Compute);
        ShaderResources::HeapReplayProof proof;
        const auto bindings = replay.resources->PrepareReplayDrawBindings(replay.recorder, moved, shaders, nullptr, proof);
        Require(bindings != nullptr && bindings->addressOffsets && proof.overlay, "heap stage replay: a moved read-only buffer did not give the draw an overlay");
        RequireMore(overlays, ReplayEvent::RunsSkipped, "a draw with an overlay built the template's descriptor run");
        Require(Access::RunOffset(*replay.resources, ShaderStage::Compute) == run, "heap stage replay: a draw whose overlay follows rewrote the template's descriptor run index");
        replay.resources->ConfirmHeapReplay(replay.recorder, shaders, bindings, proof);
        RequireSame(missing, ReplayEvent::OverlayMissing, "a draw with its predicted overlay rebuilt the template's descriptor run");
        Require(bindings->heapOffsets[0] == Access::DataOffset(*replay.resources, ShaderStage::Compute), "heap stage replay: the overlay does not carry the stage data offset the draw pushes");
        Require(bindings->heapOffsets[1] != Access::RunOffset(*replay.resources, ShaderStage::Compute), "heap stage replay: the overlay pushes the template's descriptor run instead of its own");
        replay.twin->RefreshHeapStage(shaders);
        const auto reference = replay.twin->PrepareDrawBindings(replay.recorder, moved, false, nullptr);
        Require(reference != nullptr && reference->addressOffsets, "heap stage replay: today's bindings for the moved buffer have no overlay");
        Require(reference->snapshots.size() == bindings->snapshots.size(), "heap stage replay: the replay draw and today's draw took different snapshots");
        for (std::size_t index = 0; index < bindings->snapshots.size(); ++index) {
            const auto& found = bindings->snapshots[index];
            const auto& expected = reference->snapshots[index];
            Require(found.address == expected.address && found.bytes == expected.bytes, "heap stage replay: the replay draw's snapshot of the moved buffer covers other memory than today's");
            Require(std::memcmp(found.buffer->Bytes().data() + found.offset, expected.buffer->Bytes().data() + expected.offset, found.bytes) == 0, "heap stage replay: the replay draw's snapshot of the moved buffer holds other bytes than today's");
        }
        Require(bindings->addressOverrides.size() == reference->addressOverrides.size(), "heap stage replay: the replay draw's overlay overrides other allocations than today's");
        replay.Check("a draw whose moved buffer gives it an overlay", true);
    } else {
        Require(false, "heap stage replay: the test program's input buffer is not a read-only guest allocation, so the overlay path cannot be exercised");
    }

    const auto beforePass = Count(ReplayEvent::SubmitBeforePass);
    replay.Draw(first, "a draw whose batch was submitted between the span proof and the pass", true);
    RequireMore(beforePass, ReplayEvent::SubmitBeforePass, "a submit between the span proof and the pass did not send the draw to today's refresh");
    replay.Draw(first, "a replay draw after the submit fallback");

    if (level != 0) {
        Require(AgcDriver::Graphics::ReplayChecks() != 0, "heap stage replay: the verify mode checked nothing");
        for (std::size_t field = 0; field < static_cast<std::size_t>(ReplayField::Count); ++field) {
            Require(AgcDriver::Graphics::ReplayMismatches(static_cast<ReplayField>(field)) == 0, std::string("heap stage replay: the verify mode found the replay's ") + AgcDriver::Graphics::ReplayFieldName(static_cast<ReplayField>(field)) + " wrong on correct draws");
        }
    }

    {
        replay.Draw(first, "a draw before a wrong overlay prediction");
        const auto shader = replay.Shader(first);
        const std::span<const CompiledShader> shaders(&shader, 1);
        const auto missing = Count(ReplayEvent::OverlayMissing);
        const auto before = Mismatches();
        const auto proof = replay.resources->ReplayHeapStages(shaders, true);
        bool thrown = false;
        try {
            replay.resources->ConfirmHeapReplay(replay.recorder, shaders, nullptr, proof);
        } catch (const std::runtime_error& error) {
            thrown = true;
            Require(std::string_view(error.what()).find("descriptor runs") != std::string_view::npos, std::string("heap stage replay: the strict verify mode failed a wrong overlay prediction on something other than the descriptor runs: ") + error.what());
        }
        RequireMore(missing, ReplayEvent::OverlayMissing, "a predicted overlay the draw does not have did not rebuild the template's descriptor run");
        Require(thrown == (level != 0 && strict), "heap stage replay: a wrong overlay prediction did not fail the draw exactly in the strict verify modes");
        Require((Mismatches() > before) == (level != 0), "heap stage replay: the verify mode did not report a wrong overlay prediction");
        if (!thrown) {
            replay.twin->RefreshHeapStage(shaders);
            replay.Check("a draw whose predicted overlay was missing");
        }
    }

    if (mode == "replay") {
        AgcDriver::ThreadReport::Phase();
        AgcDriver::ThreadReport::FlushForTest();
        std::puts("heap stage replay record tests passed");
        return 0;
    }
    replay.Draw(first, "a draw before the record is damaged");
    auto& record = Access::Block(*replay.resources, ShaderStage::Compute);
    record[record.size() - 1u] ^= 0x01000000u;
    const auto shader = replay.Shader(first);
    const std::span<const CompiledShader> shaders(&shader, 1);
    const auto proof = replay.resources->ReplayHeapStages(shaders, false);
    bool thrown = false;
    try {
        replay.resources->ConfirmHeapReplay(replay.recorder, shaders, nullptr, proof);
    } catch (const std::runtime_error& error) {
        thrown = true;
        Require(std::string_view(error.what()).find("stage data") != std::string_view::npos, std::string("heap stage replay: the strict verify mode failed on something other than the damaged stage data: ") + error.what());
    }
    Require(AgcDriver::Graphics::ReplayMismatches(ReplayField::Block) != 0, "heap stage replay: the verify mode did not see the damaged stage data");
    if (level == 2) {
        Require(thrown, "heap stage replay: the strict verify mode did not stop the draw with damaged stage data");
        std::puts("heap stage replay checked verify tests passed");
        return 0;
    }
    Require(!thrown, "heap stage replay: the shadow verify mode stopped a draw");
    replay.twin->RefreshHeapStage(shaders);
    replay.Check("the shadow verify mode's committed draw after the record was damaged");
    std::puts("heap stage replay shadow verify tests passed");
    return 0;
}

}

int main(int argc, char** argv) {
    const std::string_view mode = argc > 1 ? std::string_view(argv[1]) : std::string_view("replay");
    if (mode != "replay" && mode != "checked" && mode != "shadow") {
        std::cerr << "usage: heap stage replay test [replay|checked|shadow]\n";
        return 2;
    }
    SetTestEnvironment("APS5_GPU_HEAP", "1");
    if (mode == "checked") {
        SetTestEnvironment("APS5_VERIFY_REPLAY_HEAP_STAGE", "2");
        SetTestEnvironment("APS5_VERIFY_REPLAY_STRICT", "1");
    } else if (mode == "shadow") {
        SetTestEnvironment("APS5_VERIFY_REPLAY_HEAP_STAGE", "1");
        SetTestEnvironment("APS5_VERIFY_REPLAY_STRICT", "");
    }
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (SkipWithoutDescriptorHeap()) return VulkanTestSkipped;
        GuestBlock block(BlockBytes, true);
        GuestBlock surfaces(SurfaceBlockBytes, false);
        Prepare(block, surfaces);
        const auto result = Run(*device, block, mode);
        AgcDriver::Graphics::SetHeapReplayRefreshHookForTest(nullptr);
        return result;
    } catch (const std::exception& error) {
        AgcDriver::Graphics::SetHeapReplayRefreshHookForTest(nullptr);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
