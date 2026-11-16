#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DrawFinisher;
using AgcDriver::Graphics::RuntimeSwitch;
namespace RuntimeSwitches = AgcDriver::Graphics::RuntimeSwitches;

constexpr std::uint32_t Threads = 8;
constexpr std::uint32_t Side = 16;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t TileR64KBX = 0x1b;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t OutputHandle = 7;
constexpr std::size_t BlockBytes = 0x20000;
constexpr std::uint32_t Untouched = 0xdeadbeefu;
constexpr std::uint32_t Written = 0x12345678u;
constexpr std::uint32_t Frames = 6;
constexpr std::uint32_t BlockedFrame = 2;
constexpr std::uint32_t EopFrames = 4;
constexpr std::uint32_t ToggledFrames = 4;
constexpr std::int64_t EopBase = 100;
constexpr std::int64_t ToggledBase = 200;
constexpr std::uint32_t SuspendFrames = 6;
constexpr std::uint32_t SuspendBlocked = 2;
constexpr std::int64_t SuspendBase = 300;
constexpr std::uint32_t IndirectFrames = 4;
constexpr std::uint32_t IndirectBlocked = 1;
constexpr std::int64_t IndirectBase = 400;
constexpr std::uint32_t IndirectStride = 0x100;
constexpr std::uint32_t HeldRounds = 4;
constexpr std::uint32_t HeldValue = 0x3f000000u;

alignas(256) constexpr std::array<std::uint32_t, 7> StoreCode{
    0x7e040300, 0x7e060280, 0x7e0202ff, Written, 0xf0200108, 0x00020102, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 9> WriterCode{0xf4080200u, 0xfa000000u, 0xbf8cc07fu, 0x7e000202u, 0x7e020203u, 0x7e040204u, 0xe07c0000u, 0x80020000u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 7> CounterCode{0xf4080200u, 0xfa000000u, 0xbf8cc07fu, 0x7e000281u, 0xe0c80000u, 0x80020000u, 0xbf810000u};

alignas(256) constexpr std::array<std::uint32_t, 16> HeldVertexCode{
    0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
};
alignas(256) constexpr std::array<std::uint32_t, 14> HeldPixelCode{
    0xf4000100u, 0xfa000000u, 0xf4080200u, 0xfa000010u, 0xbf8cc07fu, 0x7e000204u, 0x06000002u,
    0xe0700000u, 0x80020000u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u,
};
alignas(256) std::array<std::uint32_t, 8> HeldTable{};
alignas(256) std::array<std::uint32_t, 64> HeldStorage{};
alignas(256) std::array<std::uint32_t, 64> HeldCopies{};
alignas(256) std::array<std::byte, 1024> HeldColor{};
alignas(16) std::array<std::uint32_t, 4> HeldMaskPairs{0x8e, 0xf, 0x8f, 0xf};

struct ShaderHeader {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(8) ShaderHeader HeldVertexHeader{};
alignas(8) ShaderHeader HeldPixelHeader{};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t AddressOf(const volatile void* data) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

std::uint32_t Low(const volatile void* data) {
    return static_cast<std::uint32_t>(AddressOf(data));
}

std::uint32_t High(const volatile void* data) {
    return static_cast<std::uint32_t>(AddressOf(data) >> 32u);
}

bool WaitFor(const std::function<bool()>& ready, std::chrono::milliseconds limit = std::chrono::milliseconds(10000)) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!ready()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

class Block {
public:
    Block() : storage((BlockBytes + 0x10000u) / 4u) {
        words = reinterpret_cast<std::uint32_t*>((reinterpret_cast<std::uintptr_t>(storage.data()) + 0xffffu) & ~std::uintptr_t{0xffffu});
        std::fill(words, words + BlockBytes / 4u, Untouched);
        GuestAllocations::Mutation mutation;
        mutation.Add(words, BlockBytes, true, true, true);
    }
    ~Block() {
        GuestAllocations::Mutation mutation;
        mutation.Remove(words);
    }
    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;

    std::uint32_t* words = nullptr;

private:
    std::vector<std::uint32_t> storage;
};

std::array<std::uint32_t, 8> TextureDescriptor(const std::uint32_t* texels) {
    const auto address = AddressOf(texels);
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format32UInt << 20u) | (((Side - 1u) & 3u) << 30u),
        ((Side - 1u) >> 2u) | ((Side - 1u) << 14u),
        0xfacu | (TileR64KBX << 20u) | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

void AppendStore(std::vector<std::uint32_t>& words, const std::uint32_t* texels) {
    std::vector<std::uint32_t> userData(16, 0u);
    const auto texture = TextureDescriptor(texels);
    std::copy(texture.begin(), texture.end(), userData.begin() + 8);
    const auto registers = [&](const std::uint32_t first, const std::span<const std::uint32_t> values) {
        words.push_back(0xc0007600u | (static_cast<std::uint32_t>(values.size()) << 16u));
        words.push_back(first);
        words.insert(words.end(), values.begin(), values.end());
    };
    const std::array<std::uint32_t, 3> shape{Threads, 1, 1};
    registers(0x207, shape);
    const std::array<std::uint32_t, 2> program{static_cast<std::uint32_t>(AddressOf(StoreCode.data()) >> 8u), static_cast<std::uint32_t>(AddressOf(StoreCode.data()) >> 40u)};
    registers(0x20c, program);
    const std::array<std::uint32_t, 1> resource{static_cast<std::uint32_t>(userData.size()) << 1u};
    registers(0x213, resource);
    registers(0x240, userData);
    words.insert(words.end(), {0xc0031500u, 1, 1, 1, 0x41u});
}

void AppendWrite(std::vector<std::uint32_t>& words, const volatile std::uint32_t* target, std::uint32_t value) {
    words.insert(words.end(), {0xc0033700u, 0x00100200u, Low(target), High(target), value});
}

void AppendDmaWrite(std::vector<std::uint32_t>& words, const volatile std::uint32_t* target, std::uint32_t value) {
    words.insert(words.end(), {0xc0055000u, 2u << 29u, value, 0u, Low(target), High(target), 4u});
}

void AppendPacket(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::array<std::uint32_t, 4> BufferDescriptor(const volatile void* data, std::uint32_t bytes) {
    return {Low(data), High(data) & 0xffffu, bytes, 0x31016facu};
}

void AppendCompute(std::vector<std::uint32_t>& words, const std::uint32_t* code, std::initializer_list<std::uint32_t> userData) {
    const auto registers = [&](const std::uint32_t first, const std::span<const std::uint32_t> values) {
        words.push_back(0xc0007600u | (static_cast<std::uint32_t>(values.size()) << 16u));
        words.push_back(first);
        words.insert(words.end(), values.begin(), values.end());
    };
    const std::array<std::uint32_t, 2> program{static_cast<std::uint32_t>(AddressOf(code) >> 8u), static_cast<std::uint32_t>(AddressOf(code) >> 40u) & 0xffu};
    registers(0x20c, program);
    const std::array<std::uint32_t, 3> shape{1, 1, 1};
    registers(0x207, shape);
    const std::array<std::uint32_t, 2> resources{0u, static_cast<std::uint32_t>(userData.size()) << 1u};
    registers(0x212, resources);
    const std::vector<std::uint32_t> words32(userData);
    registers(0x240, words32);
}

void AppendDmaCopy(std::vector<std::uint32_t>& words, const volatile std::uint32_t* source, const volatile std::uint32_t* target, std::uint32_t bytes) {
    words.insert(words.end(), {0xc0055000u, 0x60000000u, Low(source), High(source), Low(target), High(target), bytes});
}

void AppendList(std::vector<std::uint32_t>& words, std::uint32_t opcode, const std::uint32_t* list, std::uint32_t mode, std::uint32_t count) {
    AppendPacket(words, opcode, {Low(list), High(list), mode, count});
}

void AppendFlip(std::vector<std::uint32_t>& words, std::int64_t argument) {
    words.insert(words.end(), {AgcDriver::FlipPacketHeader, OutputHandle, 0xfffffffeu, 1, static_cast<std::uint32_t>(argument), static_cast<std::uint32_t>(static_cast<std::uint64_t>(argument) >> 32u)});
}

void AppendRenderingWait(std::vector<std::uint32_t>& words) {
    words.insert(words.end(), {AgcDriver::RenderingWaitPacketHeader, OutputHandle, 0, 0});
}

void AppendWaitEqual(std::vector<std::uint32_t>& words, const volatile std::uint32_t* target, std::uint32_t value) {
    words.insert(words.end(), {0xc0053c00u, 0x13u, Low(target), High(target), value, 0xffffffffu, 0x19u});
}

void AppendEopFlip(std::vector<std::uint32_t>& words, const volatile std::uint64_t* label, std::uint32_t value, std::uint32_t token) {
    words.insert(words.end(), {0xc0064900u, 0x06200504u, 0x42010000u, Low(label), High(label), value, 0u, token});
}

void Submit(std::vector<std::uint32_t>& words) {
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    AgcDriver::Submit(&packet, 0);
}

struct Presented {
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::int64_t> order;
    std::int64_t blocked = -1;
    bool blockedEntered = false;
    bool released = false;
};

class Request final : public AgcDriver::IFlipRequest {
public:
    Request(std::shared_ptr<Presented> presented, std::int64_t argument) : presented(std::move(presented)), argument(argument) {}
    void GpuReady(const std::shared_ptr<AgcDriver::FrameTiming>&) override {
        std::unique_lock lock(presented->mutex);
        presented->order.push_back(argument);
        if (argument == presented->blocked) {
            presented->blockedEntered = true;
            presented->changed.notify_all();
            presented->changed.wait_for(lock, std::chrono::seconds(20), [&] { return presented->released; });
        }
        presented->changed.notify_all();
    }
    void Fail(std::exception_ptr) noexcept override {}

private:
    std::shared_ptr<Presented> presented;
    std::int64_t argument;
};

struct GateState {
    std::mutex mutex;
    std::condition_variable changed;
    std::uint32_t captured = 0;
    std::uint32_t closed = ~0u;
    bool entered = false;
    bool open = false;
};

class RenderingGate final : public AgcDriver::IRenderingWait {
public:
    RenderingGate(std::shared_ptr<GateState> state, std::uint32_t ordinal) : state(std::move(state)), ordinal(ordinal) {}
    void Wait() override {
        std::unique_lock lock(state->mutex);
        if (ordinal != state->closed) return;
        state->entered = true;
        state->changed.notify_all();
        state->changed.wait_for(lock, std::chrono::seconds(20), [&] { return state->open; });
    }

private:
    std::shared_ptr<GateState> state;
    std::uint32_t ordinal;
};

class Output final : public AgcDriver::IVideoOutput {
public:
    std::shared_ptr<AgcDriver::IFlipRequest> Reserve(const AgcDriver::FlipInfo& info) override { return std::make_shared<Request>(presented, info.argument); }
    std::shared_ptr<AgcDriver::IRenderingWait> CaptureRenderingWait(std::uint32_t) override {
        std::lock_guard lock(gate->mutex);
        return std::make_shared<RenderingGate>(gate, gate->captured++);
    }
    void Fail(std::exception_ptr) noexcept override {}

    std::shared_ptr<Presented> presented = std::make_shared<Presented>();
    std::shared_ptr<GateState> gate = std::make_shared<GateState>();
};

std::uint32_t Load(const volatile std::uint32_t* word) {
    return *word;
}

bool Stored(const std::uint32_t* texels, std::size_t bytes) {
    const volatile std::uint32_t* cpu = texels;
    for (std::size_t index = 0; index < bytes / 4u; ++index) {
        if (cpu[index] == Written) return true;
    }
    return false;
}

void RequireOrder(const Output& output, std::int64_t first, std::uint32_t count, const char* what) {
    std::vector<std::int64_t> seen;
    {
        std::lock_guard lock(output.presented->mutex);
        for (const auto argument : output.presented->order) {
            if (argument >= first && argument < first + count) seen.push_back(argument);
        }
    }
    Require(seen.size() == count, std::string(what) + ": " + std::to_string(seen.size()) + " of " + std::to_string(count) + " flips reached the output");
    for (std::uint32_t index = 0; index < count; ++index) Require(seen[index] == first + index, std::string(what) + ": flip " + std::to_string(seen[index]) + " reached the output in turn " + std::to_string(index));
}

void TestFlipsBeside(Output& output, std::uint32_t* labels, std::size_t imageBytes, std::vector<std::unique_ptr<Block>>& images) {
    const bool flow = DrawFinisher::FrameFlowWanted();
    for (std::uint32_t frame = 0; frame < Frames; ++frame) images.push_back(std::make_unique<Block>());
    volatile std::uint32_t* drawn = labels;
    volatile std::uint32_t* after = labels + 16;
    volatile std::uint32_t* moved = labels + 32;
    for (std::uint32_t frame = 0; frame < Frames; ++frame) {
        drawn[frame] = 0;
        after[frame] = 0;
        moved[frame] = 0;
    }
    {
        std::lock_guard lock(output.presented->mutex);
        output.presented->blocked = BlockedFrame;
        output.presented->blockedEntered = false;
        output.presented->released = false;
    }
    std::uint32_t firstGate = 0;
    {
        std::lock_guard lock(output.gate->mutex);
        firstGate = output.gate->captured;
        output.gate->closed = firstGate + BlockedFrame;
        output.gate->entered = false;
        output.gate->open = false;
    }
    for (std::uint32_t frame = 0; frame < Frames; ++frame) {
        std::vector<std::uint32_t> words;
        AppendStore(words, images[frame]->words);
        AppendWrite(words, drawn + frame, frame + 1);
        AppendDmaWrite(words, moved + frame, frame + 1);
        AppendFlip(words, frame);
        Submit(words);
        std::vector<std::uint32_t> next;
        AppendRenderingWait(next);
        AppendWrite(next, after + frame, frame + 1);
        Submit(next);
        if (frame != BlockedFrame) continue;
        {
            std::unique_lock lock(output.presented->mutex);
            Require(output.presented->changed.wait_for(lock, std::chrono::seconds(10), [&] { return output.presented->blockedEntered; }), "the flip of frame " + std::to_string(frame) + " never reached the output");
        }
        bool enteredWhileHeld = false;
        {
            std::unique_lock lock(output.gate->mutex);
            enteredWhileHeld = output.gate->changed.wait_for(lock, std::chrono::milliseconds(flow ? 10000 : 500), [&] { return output.gate->entered; });
        }
        if (flow) Require(enteredWhileHeld, "with APS5_FRAME_FLOW the queue worker did not reach the rendering wait after the flip of frame " + std::to_string(frame) + " while the output held that flip: the flip still drained the pipeline");
        else Require(!enteredWhileHeld, "without APS5_FRAME_FLOW the queue worker reached the rendering wait after the flip of frame " + std::to_string(frame) + " while the output still held that flip");
        {
            std::lock_guard lock(output.presented->mutex);
            output.presented->released = true;
        }
        output.presented->changed.notify_all();
        {
            std::unique_lock lock(output.gate->mutex);
            Require(output.gate->changed.wait_for(lock, std::chrono::seconds(10), [&] { return output.gate->entered; }), "the queue worker never reached the rendering wait after the flip of frame " + std::to_string(frame));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        Require(Load(after + frame) == 0, "a label after the rendering wait of frame " + std::to_string(frame) + " landed while the wait was still blocked");
        {
            std::lock_guard lock(output.gate->mutex);
            output.gate->open = true;
        }
        output.gate->changed.notify_all();
    }
    AgcDriverWaitIdle_nid_postfix();
    for (std::uint32_t frame = 0; frame < Frames; ++frame) {
        Require(WaitFor([&] { return Load(drawn + frame) == frame + 1 && Load(after + frame) == frame + 1 && Load(moved + frame) == frame + 1; }), "the labels and the DMA write of frame " + std::to_string(frame) + " did not land: " + std::to_string(Load(drawn + frame)) + ", " + std::to_string(Load(after + frame)) + ", " + std::to_string(Load(moved + frame)));
        Require(WaitFor([&] { return Stored(images[frame]->words, imageBytes); }), "the image the dispatch of frame " + std::to_string(frame) + " wrote before its flip never reached guest memory");
    }
    RequireOrder(output, 0, Frames, "flip packets beside the pipeline");
    std::printf("frame flow: %u flips with the switch %s, rendering wait %s while the output held a flip\n", Frames, flow ? "in effect" : "not in effect", flow ? "reached" : "not reached");
}

void TestEopFlips(Output& output, std::uint64_t* labels) {
    volatile std::uint64_t* flipped = labels;
    volatile std::uint32_t* before = reinterpret_cast<volatile std::uint32_t*>(labels + 16);
    volatile std::uint32_t* behind = reinterpret_cast<volatile std::uint32_t*>(labels + 32);
    for (std::uint32_t frame = 0; frame < EopFrames; ++frame) {
        flipped[frame] = 0;
        before[frame] = 0;
        behind[frame] = 0;
    }
    for (std::uint32_t frame = 0; frame < EopFrames; ++frame) {
        const auto token = AgcDriverReserveEopFlip_nid_postfix(AgcDriver::FlipInfo{OutputHandle, 0, 1, EopBase + frame});
        Require(token != 0, "the EOP flip of frame " + std::to_string(frame) + " was not reserved");
        std::vector<std::uint32_t> words;
        AppendWrite(words, before + frame, frame + 1);
        AppendEopFlip(words, flipped + frame, frame + 1, token);
        Submit(words);
        std::vector<std::uint32_t> next;
        AppendWaitEqual(next, reinterpret_cast<volatile std::uint32_t*>(flipped + frame), frame + 1);
        AppendWrite(next, behind + frame, frame + 1);
        Submit(next);
    }
    AgcDriverWaitIdle_nid_postfix();
    for (std::uint32_t frame = 0; frame < EopFrames; ++frame) {
        Require(WaitFor([&] { return flipped[frame] == frame + 1 && Load(before + frame) == frame + 1 && Load(behind + frame) == frame + 1; }), "the EOP flip label of frame " + std::to_string(frame) + " or a label around it did not land");
    }
    RequireOrder(output, EopBase, EopFrames, "EOP flips");
}

void TestToggled(Output& output, std::uint32_t* labels) {
    const bool initial = RuntimeSwitches::Set(RuntimeSwitch::FrameFlow);
    volatile std::uint32_t* drawn = labels;
    volatile std::uint32_t* after = labels + 16;
    for (std::uint32_t frame = 0; frame < ToggledFrames; ++frame) {
        drawn[frame] = 0;
        after[frame] = 0;
    }
    {
        std::lock_guard lock(output.presented->mutex);
        output.presented->blocked = -1;
    }
    {
        std::lock_guard lock(output.gate->mutex);
        output.gate->closed = ~0u;
    }
    for (std::uint32_t frame = 0; frame < ToggledFrames; ++frame) {
        RuntimeSwitches::Assign(RuntimeSwitch::FrameFlow, frame % 2 == 0);
        std::vector<std::uint32_t> words;
        AppendWrite(words, drawn + frame, frame + 1);
        AppendFlip(words, ToggledBase + frame);
        AppendRenderingWait(words);
        AppendWrite(words, after + frame, frame + 1);
        Submit(words);
    }
    AgcDriverWaitIdle_nid_postfix();
    RuntimeSwitches::Assign(RuntimeSwitch::FrameFlow, initial);
    for (std::uint32_t frame = 0; frame < ToggledFrames; ++frame) {
        Require(WaitFor([&] { return Load(drawn + frame) == frame + 1 && Load(after + frame) == frame + 1; }), "a label of toggled frame " + std::to_string(frame) + " did not land");
    }
    RequireOrder(output, ToggledBase, ToggledFrames, "flips while the switch toggled between frames");
}

void TestSuspendPoints(Output& output, std::uint32_t* labels, std::size_t imageBytes, std::vector<std::unique_ptr<Block>>& images) {
    const bool ordered = DrawFinisher::OrderedSyncsWanted();
    const auto firstImage = images.size();
    for (std::uint32_t frame = 0; frame < SuspendFrames; ++frame) images.push_back(std::make_unique<Block>());
    volatile std::uint32_t* drawn = labels;
    volatile std::uint32_t* after = labels + 16;
    for (std::uint32_t frame = 0; frame < SuspendFrames; ++frame) {
        drawn[frame] = 0;
        after[frame] = 0;
    }
    {
        std::lock_guard lock(output.presented->mutex);
        output.presented->blocked = SuspendBase + SuspendBlocked;
        output.presented->blockedEntered = false;
        output.presented->released = false;
    }
    {
        std::lock_guard lock(output.gate->mutex);
        output.gate->closed = output.gate->captured;
        output.gate->entered = false;
        output.gate->open = false;
    }
    for (std::uint32_t frame = 0; frame < SuspendFrames; ++frame) {
        const bool behind = frame == SuspendBlocked + 1;
        std::vector<std::uint32_t> words;
        if (!behind) AppendStore(words, images[firstImage + frame]->words);
        AppendWrite(words, drawn + frame, frame + 1);
        Submit(words);
        AgcDriverSuspendPoint_nid_postfix();
        if (behind) {
            std::vector<std::uint32_t> waiting;
            AppendRenderingWait(waiting);
            AppendWrite(waiting, after + frame, frame + 1);
            Submit(waiting);
            bool enteredWhileHeld = false;
            {
                std::unique_lock lock(output.gate->mutex);
                enteredWhileHeld = output.gate->changed.wait_for(lock, std::chrono::milliseconds(ordered ? 10000 : 200), [&] { return output.gate->entered; });
            }
            if (ordered) Require(enteredWhileHeld, "with APS5_ORDERED_SYNCS the queue worker did not reach the rendering wait behind the suspend point of frame " + std::to_string(frame) + " while the output held the previous flip: the suspend point still drained the pipeline");
            else Require(!enteredWhileHeld, "without APS5_ORDERED_SYNCS the queue worker reached the rendering wait behind the suspend point of frame " + std::to_string(frame) + " while the output still held the previous flip");
            {
                std::lock_guard lock(output.presented->mutex);
                output.presented->released = true;
            }
            output.presented->changed.notify_all();
            {
                std::unique_lock lock(output.gate->mutex);
                Require(output.gate->changed.wait_for(lock, std::chrono::seconds(10), [&] { return output.gate->entered; }), "the queue worker never reached the rendering wait behind the suspend point of frame " + std::to_string(frame));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            Require(Load(after + frame) == 0, "a label after the rendering wait behind the suspend point of frame " + std::to_string(frame) + " landed while the wait was still blocked");
            {
                std::lock_guard lock(output.gate->mutex);
                output.gate->open = true;
            }
            output.gate->changed.notify_all();
        }
        std::vector<std::uint32_t> flip;
        AppendFlip(flip, SuspendBase + frame);
        Submit(flip);
        if (frame != SuspendBlocked) continue;
        std::unique_lock lock(output.presented->mutex);
        Require(output.presented->changed.wait_for(lock, std::chrono::seconds(10), [&] { return output.presented->blockedEntered; }), "the flip of frame " + std::to_string(frame) + " behind a suspend point never reached the output");
    }
    AgcDriverWaitIdle_nid_postfix();
    std::uint64_t accepted = 0;
    std::uint64_t completed = 0;
    AgcDriverSerialProgress_nid_postfix(&accepted, &completed);
    Require(completed == accepted, "after the frames with suspend points the driver completed " + std::to_string(completed) + " of " + std::to_string(accepted) + " accepted serials");
    for (std::uint32_t frame = 0; frame < SuspendFrames; ++frame) {
        Require(WaitFor([&] { return Load(drawn + frame) == frame + 1; }), "the label of frame " + std::to_string(frame) + " before its suspend point did not land: " + std::to_string(Load(drawn + frame)));
        if (frame == SuspendBlocked + 1) continue;
        Require(WaitFor([&] { return Stored(images[firstImage + frame]->words, imageBytes); }), "the image the dispatch of frame " + std::to_string(frame) + " wrote before its suspend point never reached guest memory");
    }
    Require(WaitFor([&] { return Load(after + SuspendBlocked + 1) == SuspendBlocked + 2; }), "the label after the rendering wait behind a suspend point did not land");
    RequireOrder(output, SuspendBase, SuspendFrames, "flips with a suspend point in every frame");
    std::printf("frame flow: %u frames with a suspend point each, ordered syncs %s\n", SuspendFrames, ordered ? "in effect" : "not in effect");
}

void TestIndirectDispatches(Output& output, std::uint32_t* block) {
    const bool ordered = DrawFinisher::OrderedSyncsWanted();
    const bool beside = ordered && RuntimeSwitches::Set(RuntimeSwitch::PipelineDispatchMisses);
    const auto at = [&](std::uint32_t frame, std::uint32_t word) { return block + frame * IndirectStride + word; };
    const auto arguments = [&](std::uint32_t frame) { return at(frame, 0); };
    const auto argumentsDescriptor = [&](std::uint32_t frame) { return at(frame, 16); };
    const auto counter = [&](std::uint32_t frame) { return at(frame, 32); };
    const auto counterDescriptor = [&](std::uint32_t frame) { return at(frame, 48); };
    const auto copied = [&](std::uint32_t frame) { return at(frame, 64); };
    const auto done = [&](std::uint32_t frame) { return at(frame, 80); };
    const auto after = [&](std::uint32_t frame) { return at(frame, 96); };
    for (std::uint32_t frame = 0; frame < IndirectFrames; ++frame) {
        std::fill(at(frame, 0), at(frame, IndirectStride), 0u);
        const auto argumentsWords = BufferDescriptor(arguments(frame), 16);
        std::copy(argumentsWords.begin(), argumentsWords.end(), argumentsDescriptor(frame));
        const auto counterWords = BufferDescriptor(counter(frame), 16);
        std::copy(counterWords.begin(), counterWords.end(), counterDescriptor(frame));
    }
    {
        std::lock_guard lock(output.presented->mutex);
        output.presented->blocked = IndirectBase + IndirectBlocked;
        output.presented->blockedEntered = false;
        output.presented->released = false;
    }
    {
        std::lock_guard lock(output.gate->mutex);
        output.gate->closed = output.gate->captured;
        output.gate->entered = false;
        output.gate->open = false;
    }
    for (std::uint32_t frame = 0; frame < IndirectFrames; ++frame) {
        const bool behind = frame == IndirectBlocked + 1;
        std::vector<std::uint32_t> words;
        AppendCompute(words, WriterCode.data(), {Low(argumentsDescriptor(frame)), High(argumentsDescriptor(frame)), frame + 1, 1u, 1u});
        words.insert(words.end(), {0xc0031500u, 1, 1, 1, 0x8041u});
        AppendCompute(words, CounterCode.data(), {Low(counterDescriptor(frame)), High(counterDescriptor(frame))});
        words.insert(words.end(), {0xc0021600u, Low(arguments(frame)), High(arguments(frame)), 0x8041u});
        if (behind) {
            AppendRenderingWait(words);
            AppendWrite(words, after(frame), frame + 1);
            Submit(words);
            words.clear();
            bool enteredWhileHeld = false;
            {
                std::unique_lock lock(output.gate->mutex);
                enteredWhileHeld = output.gate->changed.wait_for(lock, std::chrono::milliseconds(beside ? 10000 : 200), [&] { return output.gate->entered; });
            }
            if (beside) Require(enteredWhileHeld, "with APS5_ORDERED_SYNCS the queue worker did not reach the rendering wait after the DISPATCH_INDIRECT of frame " + std::to_string(frame) + " while the output held the previous flip: the indirect dispatch still drained the pipeline");
            else if (!ordered) Require(!enteredWhileHeld, "without APS5_ORDERED_SYNCS the queue worker reached the rendering wait after the DISPATCH_INDIRECT of frame " + std::to_string(frame) + " while the output still held the previous flip");
            {
                std::lock_guard lock(output.presented->mutex);
                output.presented->released = true;
            }
            output.presented->changed.notify_all();
            {
                std::unique_lock lock(output.gate->mutex);
                Require(output.gate->changed.wait_for(lock, std::chrono::seconds(10), [&] { return output.gate->entered; }), "the queue worker never reached the rendering wait after the DISPATCH_INDIRECT of frame " + std::to_string(frame));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            Require(Load(after(frame)) == 0, "a label after the rendering wait behind the DISPATCH_INDIRECT of frame " + std::to_string(frame) + " landed while the wait was still blocked");
            {
                std::lock_guard lock(output.gate->mutex);
                output.gate->open = true;
            }
            output.gate->changed.notify_all();
        }
        AppendDmaCopy(words, counter(frame), copied(frame), 4);
        AppendWrite(words, done(frame), frame + 1);
        AppendFlip(words, IndirectBase + frame);
        Submit(words);
        if (frame != IndirectBlocked) continue;
        std::unique_lock lock(output.presented->mutex);
        Require(output.presented->changed.wait_for(lock, std::chrono::seconds(10), [&] { return output.presented->blockedEntered; }), "the flip of frame " + std::to_string(frame) + " after an indirect dispatch never reached the output");
    }
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::FlushGpuWrites(AddressOf(block), static_cast<std::size_t>(IndirectFrames) * IndirectStride * sizeof(std::uint32_t));
    for (std::uint32_t frame = 0; frame < IndirectFrames; ++frame) {
        Require(WaitFor([&] { return Load(arguments(frame)) == frame + 1 && Load(arguments(frame) + 1) == 1 && Load(arguments(frame) + 2) == 1; }), "the dispatch of frame " + std::to_string(frame) + " did not write the indirect arguments: " + std::to_string(Load(arguments(frame))));
        Require(WaitFor([&] { return Load(counter(frame)) == frame + 1; }), "the DISPATCH_INDIRECT of frame " + std::to_string(frame) + " ran " + std::to_string(Load(counter(frame))) + " groups, expected " + std::to_string(frame + 1));
        Require(WaitFor([&] { return Load(copied(frame)) == frame + 1; }), "the DMA copy after the DISPATCH_INDIRECT of frame " + std::to_string(frame) + " read " + std::to_string(Load(copied(frame))) + ", expected " + std::to_string(frame + 1));
        Require(WaitFor([&] { return Load(done(frame)) == frame + 1; }), "the label of frame " + std::to_string(frame) + " after its indirect dispatch did not land");
    }
    Require(WaitFor([&] { return Load(after(IndirectBlocked + 1)) == IndirectBlocked + 2; }), "the label after the rendering wait behind an indirect dispatch did not land");
    RequireOrder(output, IndirectBase, IndirectFrames, "flips after indirect dispatches");
    std::printf("frame flow: %u frames of indirect dispatches, ordered syncs %s, rendering wait %s while the output held a flip\n", IndirectFrames, ordered ? "in effect" : "not in effect", beside ? "reached" : "not required");
}

void RegisterShader(ShaderHeader& header, const std::uint32_t* code, std::size_t bytes, std::uint8_t type) {
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(ShaderHeader);
    header.shader.shader_size = static_cast<std::uint32_t>(bytes);
    header.shader.code = code;
    header.shader.type = type;
    header.shader.user_data = &header.userData;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

std::vector<std::uint32_t> HeldDrawState() {
    const auto color = AddressOf(HeldColor.data());
    const auto vertex = AddressOf(HeldVertexCode.data());
    const auto pixel = AddressOf(HeldPixelCode.data());
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> context{
        {0x2d5, 0x2000},
        {0x1b6, 0}, {0x207, 0}, {0x200, 0}, {0x203, 0x800},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
        {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28028},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, (63u << 14u) | 3u},
        {0x3b8, 0x9000000}, {0x1e0, 0},
        {0xc, 0}, {0xd, 0x40040},
        {0x81, 0x80000000}, {0x82, 0x40040},
        {0x90, 0x80000000}, {0x91, 0x40040},
        {0x94, 0x80000000}, {0x95, 0x40040},
        {0x318, static_cast<std::uint32_t>(color >> 8u)}, {0x390, static_cast<std::uint32_t>(color >> 40u)},
        {0x10f, std::bit_cast<std::uint32_t>(32.0f)}, {0x110, std::bit_cast<std::uint32_t>(32.0f)},
        {0x111, std::bit_cast<std::uint32_t>(-2.0f)}, {0x112, std::bit_cast<std::uint32_t>(2.0f)},
        {0x113, std::bit_cast<std::uint32_t>(1.0f)}, {0x114, 0},
        {0xb4, 0}, {0xb5, std::bit_cast<std::uint32_t>(1.0f)},
        {0x2fe, 0}, {0x302, 0}, {0x306, 0}, {0x30a, 0},
    };
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> shader{
        {0xc8, static_cast<std::uint32_t>(vertex >> 8u)}, {0xc9, static_cast<std::uint32_t>(vertex >> 40u)}, {0x8b, 0}, {0x8c, 0},
        {0x008, static_cast<std::uint32_t>(pixel >> 8u)}, {0x009, static_cast<std::uint32_t>(pixel >> 40u)}, {0x00b, 3u << 1u},
    };
    std::vector<std::uint32_t> words;
    AppendPacket(words, 0x79, {0x242, 4});
    for (const auto& [offset, value] : context) AppendPacket(words, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) AppendPacket(words, 0x76, {offset, value});
    const auto table = AddressOf(HeldTable.data());
    AppendPacket(words, 0x76, {0x00c, static_cast<std::uint32_t>(table)});
    AppendPacket(words, 0x76, {0x00d, static_cast<std::uint32_t>(table >> 32u)});
    AppendPacket(words, 0x76, {0x00e, std::bit_cast<std::uint32_t>(0.25f)});
    return words;
}

void TestHeldDraws(bool libraries) {
    if (!libraries) {
        std::puts("frame flow: the held draw case needs graphics pipeline libraries, skipped");
        return;
    }
    RegisterShader(HeldVertexHeader, HeldVertexCode.data(), sizeof(HeldVertexCode), 2);
    RegisterShader(HeldPixelHeader, HeldPixelCode.data(), sizeof(HeldPixelCode), 1);
    HeldTable.fill(0u);
    HeldTable[0] = std::bit_cast<std::uint32_t>(0.25f);
    const auto storage = BufferDescriptor(HeldStorage.data(), sizeof(HeldStorage));
    std::copy(storage.begin(), storage.end(), HeldTable.begin() + 4);
    HeldStorage.fill(0u);
    HeldCopies.fill(0u);
    HeldColor.fill(std::byte{0});
    const bool naming = DrawFinisher::Enabled() && DrawFinisher::Overlap() && DrawFinisher::OrderedSyncsWanted();
    const auto namedBefore = DrawFinisher::HeldNamedTotal();
    volatile std::uint32_t* copies = HeldCopies.data();
    for (std::uint32_t round = 0; round < HeldRounds; ++round) {
        auto words = HeldDrawState();
        AppendPacket(words, 0x2d, {3, 2});
        AppendList(words, 0x9f, HeldMaskPairs.data(), 0x80000000u, 2);
        AppendDmaCopy(words, HeldStorage.data(), copies + round, 4);
        AppendWrite(words, copies + 32 + round, round + 1);
        Submit(words);
        AgcDriverWaitIdle_nid_postfix();
    }
    AgcDriver::GuestMemory::FlushGpuWrites(AddressOf(HeldStorage.data()), sizeof(HeldStorage));
    AgcDriver::GuestMemory::FlushGpuWrites(AddressOf(HeldCopies.data()), sizeof(HeldCopies));
    Require(WaitFor([&] { return Load(HeldStorage.data()) == HeldValue; }), "the pixel shader's store to its storage buffer did not land: " + std::to_string(Load(HeldStorage.data())));
    for (std::uint32_t round = 0; round < HeldRounds; ++round) {
        Require(WaitFor([&] { return Load(copies + round) == HeldValue; }), "the DMA copy after the held draw of round " + std::to_string(round) + " read " + std::to_string(Load(copies + round)) + " from the storage buffer the draw writes, expected " + std::to_string(HeldValue));
        Require(WaitFor([&] { return Load(copies + 32 + round) == round + 1; }), "the label after the held draw of round " + std::to_string(round) + " did not land");
    }
    const auto named = DrawFinisher::HeldNamedTotal() - namedBefore;
    if (naming) Require(named != 0, "with APS5_ORDERED_SYNCS no held draw of " + std::to_string(HeldRounds) + " whose pixel shader writes a storage buffer was published with its writes named");
    std::printf("frame flow: %u draws whose pixel shader stores to a buffer, %llu published with named writes\n", HeldRounds, static_cast<unsigned long long>(named));
}

}

int main() {
    try {
        bool libraries = false;
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
            libraries = device->GraphicsPipelineLibraries();
        }
        Block probe;
        const auto imageBytes = AgcDriver::Graphics::DescribeSurface(AgcDriver::Graphics::DecodeTextureResource(TextureDescriptor(probe.words))).guestBytes;
        Require(imageBytes <= AgcDriver::Graphics::StorageTexture::SmallStoreBytes, "the test image is larger than an image the flip stores");
        Block labelBlock;
        std::fill(labelBlock.words, labelBlock.words + BlockBytes / 4u, 0u);
        std::vector<std::unique_ptr<Block>> images;
        auto output = std::make_shared<Output>();
        AgcDriverRegisterVideoOutput_nid_postfix(OutputHandle, output);
        TestFlipsBeside(*output, labelBlock.words, imageBytes, images);
        TestEopFlips(*output, reinterpret_cast<std::uint64_t*>(labelBlock.words + 0x1000));
        TestToggled(*output, labelBlock.words + 0x2000);
        TestSuspendPoints(*output, labelBlock.words + 0x3000, imageBytes, images);
        TestIndirectDispatches(*output, labelBlock.words + 0x4000);
        TestHeldDraws(libraries);
        AgcDriverWaitIdle_nid_postfix();
        AgcDriverUnregisterVideoOutput_nid_postfix(OutputHandle, output);
        AgcDriverShutdown_nid_postfix();
        std::puts("frame flow tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
