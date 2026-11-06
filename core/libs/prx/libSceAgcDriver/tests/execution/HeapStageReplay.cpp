#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BindlessHeap.hpp"
#include "prx/libSceAgcDriver/Graphics/include/HeapReplay.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
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
    static std::uint32_t DataOffset(const ShaderResources& resources, ShaderRecompiler::ShaderStage stage) { return resources.heapOffsets[HeapStageIndex(stage) * 2u]; }
    static std::uint32_t RunOffset(const ShaderResources& resources, ShaderRecompiler::ShaderStage stage) { return resources.heapOffsets[HeapStageIndex(stage) * 2u + 1u]; }
    static std::optional<std::size_t> GuestAllocation(const ShaderResources& resources, std::uint64_t address) {
        for (std::size_t index = 0; index < resources.allocations.size(); ++index) {
            if (resources.allocations[index].guest && !resources.allocations[index].written && resources.allocations[index].address == address) return index;
        }
        return std::nullopt;
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

const std::vector<std::uint32_t>* ShaderDataWords(const ShaderRecompiler::RecompileResult& program) {
    for (const auto& binding : program.bindings) {
        if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) return &binding.guestDescriptor;
    }
    return nullptr;
}

std::uint64_t Count(ReplayEvent event) {
    return ReplayEvents(event);
}

struct Replay {
    std::shared_ptr<ShaderResources> resources;
    std::shared_ptr<ShaderResources> twin;
    AgcDriver::Graphics::BindlessHeap& heap;
    AgcDriver::Graphics::Recorder& recorder;
    const ShaderRecompiler::RecompileResult& program;

    CompiledShader Shader(const ShaderRecompiler::HeapStageWords& words) const { return {ShaderStage::Compute, &program, 0, nullptr, &words}; }

    void Check(const char* what) const {
        auto& block = Access::Block(*resources, ShaderStage::Compute);
        Require(block == Access::Block(*twin, ShaderStage::Compute), std::string("heap stage replay: ") + what + ": the stage data differs from the one today's refresh builds");
        const auto mark = Access::DataMark(*resources, ShaderStage::Compute);
        Require(Access::DataOffset(*resources, ShaderStage::Compute) == mark.offset, std::string("heap stage replay: ") + what + ": the root's shader data offset is not the stage data span");
        Require(heap.SpanCurrent(mark), std::string("heap stage replay: ") + what + ": the stage data span is not current");
        const auto ring = heap.RingWords(mark);
        Require(std::ranges::equal(ring, block), std::string("heap stage replay: ") + what + ": the ring does not hold the stage data");
    }

    void Draw(const ShaderRecompiler::HeapStageWords& words, const char* what, bool overlay = false, bool submitBeforePass = false) {
        const auto shader = Shader(words);
        const std::span<const CompiledShader> shaders(&shader, 1);
        const auto proof = resources->ReplayHeapStages(shaders, overlay);
        if (submitBeforePass) heap.Submitted(recorder, 0u);
        resources->ConfirmHeapReplay(recorder, shaders, nullptr, proof);
        twin->RefreshHeapStage(shaders);
        Check(what);
    }
};

void RequireMore(std::uint64_t before, ReplayEvent event, const char* what) {
    Require(Count(event) > before, std::string("heap stage replay: ") + what);
}

int Run(const AgcDriver::VulkanDevice& device, const GuestBlock& block, std::string_view mode) {
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    if (!context.descriptorHeap || !context.heapBufferDescriptors) {
        std::puts("skipped, the device keeps no buffer descriptors in the shader data ring, so no stage takes the heap stage replay");
        return VulkanTestSkipped;
    }
    Require(context.recorder != nullptr, "heap stage replay: the device has no recorder");
    const auto program = Compile(device, block);
    Require(program.gpuResolvedImages && program.dataAddressPush, "heap stage replay: the program is not in the heap's GPU-resolved form");
    const auto* dataWords = ShaderDataWords(program);
    Require(dataWords != nullptr && !dataWords->empty(), "heap stage replay: the program has no shader data");
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    const CompiledShader built{ShaderStage::Compute, &program, 0};
    Replay replay{std::make_shared<ShaderResources>(context, built), std::make_shared<ShaderResources>(context, built), AgcDriver::Graphics::BindlessHeap::For(context), *context.recorder, program};
    Require(replay.resources->HeapStage(), "heap stage replay: the set was not built on the heap");
    const auto userDwords = std::min<std::size_t>(dataWords->size(), 8u);
    ShaderRecompiler::HeapStageWords first;
    first.userData.assign(dataWords->begin(), dataWords->begin() + static_cast<std::ptrdiff_t>(userDwords));

    const auto created = Count(ReplayEvent::NewRecord);
    replay.Draw(first, "the first replay draw");
    RequireMore(created, ReplayEvent::NewRecord, "the first draw did not create the template's record");

    const auto stages = Count(ReplayEvent::Stages);
    const auto proven = Count(ReplayEvent::DataProven);
    replay.Draw(first, "a repeated draw");
    RequireMore(stages, ReplayEvent::Stages, "a repeated draw did not replay its stage");
    RequireMore(proven, ReplayEvent::DataProven, "a repeated draw in the same batch did not keep its stage data span by the submit epoch");

    std::optional<std::size_t> free;
    for (std::size_t dword = 0; dword < userDwords; ++dword) {
        if (!Access::Masked(*replay.resources, ShaderStage::Compute, dword)) {
            free = dword;
            break;
        }
    }
    auto changed = first;
    if (free.has_value()) {
        changed.userData[*free] ^= 0x5a5a0000u;
        const auto patched = Count(ReplayEvent::DataPatched);
        replay.Draw(changed, "a draw with one changed user data dword");
        RequireMore(patched, ReplayEvent::DataPatched, "a changed user data dword was not patched into the stage data");
    } else {
        std::puts("every user data dword of the test program is patched by the template; the patch step is not exercised");
    }

    replay.heap.Submitted(replay.recorder, 0u);
    const auto held = Count(ReplayEvent::DataHeld);
    replay.Draw(changed, "a repeated draw after a submit");
    RequireMore(held, ReplayEvent::DataHeld, "a repeated draw after a submit did not hold its stage data span again");

    const auto writes = Count(ReplayEvent::NonReplayWrites);
    auto other = first;
    if (free.has_value()) other.userData[*free] ^= 0x00a50000u;
    {
        const auto shader = replay.Shader(other);
        replay.resources->RefreshHeapStage(std::span<const CompiledShader>(&shader, 1));
    }
    replay.Draw(other, "the first replay draw after another writer refreshed the stage");
    RequireMore(writes, ReplayEvent::NonReplayWrites, "a refresh outside the replay did not invalidate the template's record");
    replay.Draw(other, "a replay draw after the record was rebuilt");

    const auto skipped = Count(ReplayEvent::RunsSkipped);
    const auto run = Access::RunOffset(*replay.resources, ShaderStage::Compute);
    {
        const auto shader = replay.Shader(other);
        const std::span<const CompiledShader> shaders(&shader, 1);
        const auto proof = replay.resources->ReplayHeapStages(shaders, true);
        Require(Access::RunOffset(*replay.resources, ShaderStage::Compute) == run, "heap stage replay: a draw whose overlay follows rewrote the template's descriptor run index");
        replay.resources->ConfirmHeapReplay(replay.recorder, shaders, nullptr, proof);
    }
    RequireMore(skipped, ReplayEvent::RunsSkipped, "a draw whose overlay follows built the template's descriptor run");

    if (const auto input = Access::GuestAllocation(*replay.resources, block.At(InputOffset)); input.has_value() && Access::GuestAllocation(*replay.twin, block.At(InputOffset)) == input) {
        const std::array<ShaderResources::MovedBuffer, 1> moved{{{*input, block.At(InputOffset + 1024u), Lanes * 4u, {}}}};
        const auto shader = replay.Shader(other);
        const std::span<const CompiledShader> shaders(&shader, 1);
        const auto overlays = Count(ReplayEvent::RunsSkipped);
        ShaderResources::HeapReplayProof proof;
        const auto bindings = replay.resources->PrepareReplayDrawBindings(replay.recorder, moved, shaders, nullptr, proof);
        Require(bindings != nullptr && bindings->addressOffsets && proof.overlay, "heap stage replay: a moved read-only buffer did not give the draw an overlay");
        RequireMore(overlays, ReplayEvent::RunsSkipped, "a draw with an overlay built the template's descriptor run");
        replay.resources->ConfirmHeapReplay(replay.recorder, shaders, bindings, proof);
        Require(bindings->heapOffsets[0] == Access::DataOffset(*replay.resources, ShaderStage::Compute), "heap stage replay: the overlay does not carry the stage data offset the draw pushes");
        replay.twin->RefreshHeapStage(shaders);
        const auto reference = replay.twin->PrepareDrawBindings(replay.recorder, moved, false, nullptr);
        Require(reference != nullptr && reference->addressOffsets, "heap stage replay: today's bindings for the moved buffer have no overlay");
        Require(reference->snapshots.size() == bindings->snapshots.size(), "heap stage replay: the replay draw and today's draw took different snapshots");
        replay.Check("a draw whose moved buffer gives it an overlay");
    } else {
        std::puts("the test program's input buffer is not a read-only guest allocation; the overlay path is not exercised");
    }

    const auto beforePass = Count(ReplayEvent::SubmitBeforePass);
    replay.Draw(first, "a draw whose batch was submitted between the span proof and the pass", false, true);
    RequireMore(beforePass, ReplayEvent::SubmitBeforePass, "a submit between the span proof and the pass did not send the draw to today's refresh");
    replay.Draw(first, "a replay draw after the submit fallback");

    if (mode == "replay") {
        AgcDriver::ThreadReport::Phase();
        std::puts("heap stage replay record tests passed");
        return 0;
    }
    const auto level = AgcDriver::Graphics::ReplayVerifyLevel();
    Require(level == (mode == "shadow" ? 1 : 2), "heap stage replay: the verify mode does not have the level the test runs with");
    const auto checks = AgcDriver::Graphics::ReplayChecks();
    Require(checks != 0, "heap stage replay: the verify mode checked nothing");
    for (std::size_t field = 0; field < static_cast<std::size_t>(ReplayField::Count); ++field) {
        Require(AgcDriver::Graphics::ReplayMismatches(static_cast<ReplayField>(field)) == 0, std::string("heap stage replay: the verify mode found the replay's ") + AgcDriver::Graphics::ReplayFieldName(static_cast<ReplayField>(field)) + " wrong on correct draws");
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
        return Run(*device, block, mode);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
