#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
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

alignas(256) constexpr std::array<std::uint32_t, 7> StoreCode{
    0x7e040300, 0x7e060280, 0x7e0202ff, Written, 0xf0200108, 0x00020102, 0xbf810000,
};

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

}

int main() {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
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
