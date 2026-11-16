#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DirectDrawReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRecords.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/StagePlanReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadReport.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgc/Shader/include/ShaderConstants.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/WriteValidation.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
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
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

namespace {

using AgcDriver::Graphics::Require;
namespace DrawRecords = AgcDriver::DriverDetail::DrawRecords;
using Counter = DrawRecords::Counter;

constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t ColorBytes = 65536;
constexpr std::uint32_t RenderTargetTileMode = 0x1b;
constexpr std::size_t TableOffset = 0x300;
constexpr std::size_t NearOffset = 0x800;
constexpr std::size_t ShiftOffset = 0x900;
constexpr std::size_t StillOffset = 0xa00;
constexpr std::size_t FarOffset = 0xb00;
constexpr std::size_t ConstantsOffset = 0xc00;
constexpr std::size_t FetchBufferTableOffset = 0x000;
constexpr std::size_t FetchAttribTableOffset = 0x100;
constexpr std::size_t ShiftedSlotsOffset = 0x200;
constexpr std::size_t CurrentSlotsOffset = 0x400;
constexpr std::size_t InstanceSlotsOffset = 0x600;
constexpr std::uint32_t SlotRecords = 32;
constexpr std::uint32_t SlotShift = 16;
constexpr std::uint32_t OutputSlots = 32;
constexpr std::uint32_t Unwritten = 0xdeadbeefu;
constexpr std::uint32_t FetchIndexOffset = 5;
constexpr std::uint32_t FetchVertexOffset = 3;
constexpr std::uint32_t DrawsPerSubmission = 6;
constexpr std::size_t RingSlotsOffset = 0x1000;
constexpr std::size_t RingSlotBytes = 0x100;
constexpr std::size_t InnerOffset = 0x4000;
constexpr std::size_t InnerBytes = 0x100;
constexpr std::size_t RingSlots = 16;
constexpr std::size_t InnerBuffers = 8;
constexpr std::uint32_t ConstantRecords = 64;
constexpr std::uint32_t InnerRecords = 16;
constexpr std::array<std::uint32_t, 4> RawFormats{0x31016facu, 0x31014facu, 0x31015facu, 0x21016facu};
constexpr std::array<std::uint32_t, 2> SlotFormats{0x01014facu, 0x01015facu};
constexpr std::size_t TextureBlockBytes = 0x40000;
constexpr std::array<std::size_t, 2> TextureOffsets{0x0, 0x20000};
constexpr std::size_t TextureBytes = 1024;
constexpr std::uint32_t TextureUserWords = 12;
constexpr std::array<std::array<std::uint8_t, 4>, 2> Texels{{{0x10u, 0x80u, 0xc0u, 0xffu}, {0x33u, 0x66u, 0x99u, 0xffu}}};
constexpr std::array<std::uint32_t, 4> TextureVertexCounts{3u, 6u, 9u, 12u};
constexpr std::uint32_t TextureDraws = 3 * DrawsPerSubmission;

alignas(256) constexpr std::array<std::uint32_t, 16> PlainVertex{
    0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 10> ConstantPixel{0xf4000100u, 0xfa000000u, 0xbf8cc07fu, 0x7e000204u, 0x06000002u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u};

alignas(256) constexpr std::array<std::uint32_t, 40> TableVertex{
    0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e060280u, 0x7e0802f2u,
    0xf4080104u, 0xfa000000u, 0xf4080204u, 0xfa000010u, 0xf4080304u, 0xfa000020u,
    0xbf8cc07fu,
    0xe00c2000u, 0x80010a05u, 0xe00c2000u, 0x80020e05u, 0xe00c2000u, 0x80031205u,
    0xbf8c3f70u,
    0x0602030au, 0x0604050bu, 0x0606070cu, 0x0608090du,
    0x0602030eu, 0x0604050fu, 0x06060710u, 0x06080911u,
    0x06020312u, 0x06040513u, 0x06060714u, 0x06080915u,
    0xf80008cfu, 0x04030201u,
    0xbf810000u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 11> ReadPixel{0x7e1a0280u, 0xdc308000u, 0x0c02000du, 0xbf8c3f70u, 0x7e10030cu, 0x7e120280u, 0x7e140280u, 0x7e160280u, 0xf800180fu, 0x0b0a0908u, 0xbf810000u};

alignas(256) constexpr std::array<std::uint32_t, 30> FetchVertex{
    0x7e020305u, 0x7e040308u,
    0xd70f6a05u, 0x00020a10u,
    0xd70f6a08u, 0x00021011u,
    0xf4080506u, 0xfa000000u,
    0xf4080606u, 0xfa000010u,
    0x02060b08u,
    0xbf8cc07fu,
    0xe0002000u, 0x80050a03u,
    0xe0002000u, 0x80060b03u,
    0xbf8c3f70u,
    0x4a08170au, 0x34080884u,
    0x7e180301u, 0x7e1a0302u, 0x7e1c0305u, 0x7e1e0308u,
    0xbf800000u, 0xbf800000u,
    0x7e200280u, 0x7e2202f2u,
    0xf80008cfu, 0x1110100au,
    0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 16> RingPixel{
    0xf4280100u, 0xfa000010u,
    0xf4200200u, 0xfa000000u,
    0xbf8cc07fu,
    0xe0300000u, 0x80010400u,
    0xbf8c3f70u,
    0x7e000208u, 0x7e020304u, 0x7e040280u, 0x7e0602f2u,
    0xf800180fu, 0x03020100u,
    0xbf810000u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 12> SamplePixel{0x7e0002f0u, 0x7e0202f0u, 0xf09c0f08u, 0x00400000u, 0xbf8c0000u, 0xf800180fu, 0x03020100u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u};

struct Header {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(8) Header PlainVertexHeader{};
alignas(8) Header ConstantPixelHeader{};
alignas(8) Header TableVertexHeader{};
alignas(8) Header ReadPixelHeader{};
alignas(8) Header RingPixelHeader{};
alignas(8) Header SamplePixelHeader{};

alignas(256) constexpr std::array<std::uint32_t, 17> DataVertex{
    0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e060280u, 0x7e0802f2u,
    0xf4000284u, 0xfa000000u, 0xbf8cc07fu, 0x0602020au,
    0xf80008cfu, 0x04030201u, 0xbf810000u,
};

alignas(8) Header DataVertexHeader{};

void Register(Header& header, const std::uint32_t* code, std::size_t bytes, std::uint8_t type) {
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(bytes);
    header.shader.code = code;
    header.shader.type = type;
    header.shader.user_data = &header.userData;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

struct FetchHeader {
    Shader shader{};
    ShaderUserData userData{};
    std::array<std::uint16_t, ShaderRegs::AGC_DIRECT_RESOURCE_TYPE_COUNT> direct{};
    std::array<ShaderSemantic, 2> semantics{};
};

alignas(8) FetchHeader FetchVertexHeader{};

void RegisterFetchVertex() {
    auto& header = FetchVertexHeader;
    header.direct.fill(ShaderRegs::AGC_ILLEGAL_DIRECT_OFFSET);
    header.direct[static_cast<std::size_t>(ShaderRegs::AgcDirectResourceType::PtrVertexBufferTable)] = 4;
    header.direct[static_cast<std::size_t>(ShaderRegs::AgcDirectResourceType::PtrVertexAttribDescTable)] = 6;
    header.userData.direct_resource_offset = header.direct.data();
    header.userData.direct_resource_count = static_cast<std::uint16_t>(header.direct.size());
    header.semantics[0].semantic = 0;
    header.semantics[0].hardware_mapping = 10;
    header.semantics[0].size_in_elements = 1;
    header.semantics[1].semantic = 1;
    header.semantics[1].hardware_mapping = 11;
    header.semantics[1].size_in_elements = 1;
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(FetchHeader);
    header.shader.shader_size = static_cast<std::uint32_t>(sizeof(FetchVertex));
    header.shader.code = FetchVertex.data();
    header.shader.type = 2;
    header.shader.user_data = &header.userData;
    header.shader.input_semantics = header.semantics.data();
    header.shader.num_input_semantics = static_cast<std::uint32_t>(header.semantics.size());
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, 65536));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, BlockBytes, PAGE_READWRITE, BlockBytes);
#else
        constexpr std::uintptr_t alignment = 65536;
        void* mapped = mmap(nullptr, BlockBytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped != MAP_FAILED) {
            const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
            const auto aligned = (begin + alignment - 1) & ~(alignment - 1);
            if (aligned != begin) munmap(mapped, aligned - begin);
            if (aligned + BlockBytes != begin + BlockBytes + alignment) munmap(reinterpret_cast<void*>(aligned + BlockBytes), begin + alignment - aligned);
            block = reinterpret_cast<std::uint8_t*>(aligned);
            GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, BlockBytes);
        }
#endif
        Require(block != nullptr, "draw record hits: cannot allocate a guest block");
        std::memset(block, 0, BlockBytes);
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint64_t At(std::size_t offset) const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block + offset)); }
    std::uint8_t* Bytes(std::size_t offset) const { return block + offset; }

private:
    std::uint8_t* block = nullptr;
};

class TextureBlock {
public:
    TextureBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(TextureBlockBytes, 0x200000));
        Require(block != nullptr, "draw record hits: cannot reserve the texture block");
        section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(TextureBlockBytes), nullptr);
        Require(section != nullptr, "draw record hits: cannot create the texture block's backing");
        GuestArena::GuestArenaMapDirect_nid_postfix(block, TextureBlockBytes, section, 0, PAGE_READWRITE);
#else
        constexpr std::uintptr_t alignment = 65536;
        void* mapped = mmap(nullptr, TextureBlockBytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped != MAP_FAILED) {
            const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
            const auto aligned = (begin + alignment - 1) & ~(alignment - 1);
            if (aligned != begin) munmap(mapped, aligned - begin);
            if (aligned + TextureBlockBytes != begin + TextureBlockBytes + alignment) munmap(reinterpret_cast<void*>(aligned + TextureBlockBytes), begin + alignment - aligned);
            block = reinterpret_cast<std::uint8_t*>(aligned);
            GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, TextureBlockBytes);
        }
        Require(block != nullptr, "draw record hits: cannot allocate the texture block");
#endif
        std::memset(block, 0, TextureBlockBytes);
        GuestAllocations::Mutation().Add(block, TextureBlockBytes, true, true, true);
    }

    TextureBlock(const TextureBlock&) = delete;
    TextureBlock& operator=(const TextureBlock&) = delete;

    std::uint64_t At(std::size_t offset) const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block + offset)); }
    std::uint8_t* Bytes(std::size_t offset) const { return block + offset; }

    bool Armed(std::size_t offset) const {
#ifdef _WIN32
        MEMORY_BASIC_INFORMATION info{};
        return VirtualQuery(block + offset, &info, sizeof(info)) == sizeof(info) && info.Protect == PAGE_READONLY;
#else
        static_cast<void>(offset);
        return false;
#endif
    }

    bool Arm(std::size_t offset) const {
        AgcDriver::GuestMemory::CollectWritesUncached(At(0), TextureBlockBytes);
        AgcDriver::GuestMemory::CollectWritesUncached(At(0), TextureBlockBytes);
        return Armed(offset);
    }

private:
    std::uint8_t* block = nullptr;
#ifdef _WIN32
    HANDLE section = nullptr;
#endif
};

std::array<std::uint32_t, 8> TextureDescriptor(std::uint64_t address) {
    return {static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (56u << 20u) | (3u << 30u), 3u << 14u, 0xfacu | (9u << 28u), 0u, 0u, 0u, 0u};
}

using RegisterWrites = std::vector<std::pair<std::uint32_t, std::uint32_t>>;

void Append(std::vector<std::uint32_t>& commands, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    commands.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    commands.insert(commands.end(), payload);
}

using DrawStep = std::function<std::uint32_t(std::vector<std::uint32_t>&, std::uint32_t)>;

void SubmitDraws(std::uint64_t vertex, std::uint64_t pixel, const RegisterWrites& userConfig, const RegisterWrites& shaderRegisters, const GuestBlock& target, std::uint32_t draws, std::uint32_t vertices = 3, std::uint32_t blend = 0, const DrawStep& step = {}) {
    const auto color = target.At(0);
    const float scale = 32.0f;
    const RegisterWrites context{
        {0x2d5, 0x2000},
        {0x1b6, 0}, {0x207, 0}, {0x200, 0}, {0x203, 0x800},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
        {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28028},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, (63u << 14u) | 3u},
        {0x3b8, 0x9000000u | (RenderTargetTileMode << 14u)}, {0x1e0, blend},
        {0xc, 0}, {0xd, 0x40040},
        {0x81, 0x80000000}, {0x82, 0x40040},
        {0x90, 0x80000000}, {0x91, 0x40040},
        {0x94, 0x80000000}, {0x95, 0x40040},
        {0x318, static_cast<std::uint32_t>(color >> 8u)}, {0x390, static_cast<std::uint32_t>(color >> 40u)},
        {0x10f, std::bit_cast<std::uint32_t>(scale)}, {0x110, std::bit_cast<std::uint32_t>(scale)},
        {0x111, std::bit_cast<std::uint32_t>(-2.0f)}, {0x112, std::bit_cast<std::uint32_t>(2.0f)},
        {0x113, std::bit_cast<std::uint32_t>(1.0f)}, {0x114, 0},
        {0xb4, 0}, {0xb5, std::bit_cast<std::uint32_t>(1.0f)},
        {0x2fe, 0}, {0x302, 0}, {0x306, 0}, {0x30a, 0},
    };
    std::vector<std::uint32_t> commands;
    for (const auto& [offset, value] : userConfig) Append(commands, 0x79, {offset, value});
    for (const auto& [offset, value] : context) Append(commands, 0x69, {offset, value});
    Append(commands, 0x76, {0xc8, static_cast<std::uint32_t>(vertex >> 8u)});
    Append(commands, 0x76, {0xc9, static_cast<std::uint32_t>(vertex >> 40u)});
    Append(commands, 0x76, {0x008, static_cast<std::uint32_t>(pixel >> 8u), static_cast<std::uint32_t>(pixel >> 40u)});
    for (const auto& [offset, value] : shaderRegisters) Append(commands, 0x76, {offset, value});
    for (std::uint32_t draw = 0; draw < draws; ++draw) {
        const auto count = step ? step(commands, draw) : vertices;
        Append(commands, 0x2d, {count, 2});
    }
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "draw record hits: a draw submission failed");
    AgcDriverWaitIdle_nid_postfix();
}

std::vector<std::uint8_t> Image(const GuestBlock& target) {
    AgcDriver::GuestMemory::FlushGpuWrites(static_cast<std::uintptr_t>(target.At(0)), ColorBytes);
    return {target.Bytes(0), target.Bytes(0) + ColorBytes};
}

std::array<std::uint32_t, 4> VertexDescriptor(std::uint64_t address, std::uint32_t stride = 16u) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), 3u, 0x0104dfacu};
}

std::array<std::uint32_t, 4> SlotDescriptor(std::uint64_t address, std::uint32_t format = SlotFormats[0]) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (4u << 16u), SlotRecords, format};
}

std::array<std::uint32_t, 4> OutputDescriptor(std::uint64_t address) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), OutputSlots * 16u, 0x31016facu};
}

std::array<std::uint32_t, 4> RawDescriptor(std::uint64_t address, std::uint32_t records, std::uint32_t format = RawFormats[0]) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), records, format};
}

void WriteWords(const GuestBlock& block, std::size_t offset, std::array<std::uint32_t, 4> words) {
    std::memcpy(block.Bytes(offset), words.data(), sizeof(words));
}

void WriteVertices(const GuestBlock& block, std::size_t offset, float x) {
    for (std::size_t vertex = 0; vertex < 3; ++vertex) {
        const std::array<float, 4> value{x, 0.0f, 0.0f, 0.0f};
        std::memcpy(block.Bytes(offset + vertex * 16u), value.data(), sizeof(value));
    }
}

struct Blocks {
    GuestBlock vertices;
    GuestBlock data;
    GuestBlock target;
    GuestBlock fetch;
    GuestBlock output;
    GuestBlock ring;
    GuestBlock other;
    std::unique_ptr<TextureBlock> texture;
};

Blocks* Scene = nullptr;

enum class Kind { Constants, Table, Fetch, Ring, Inner, RingInner, Sized, Format, FetchFormat, Texture };

struct State {
    Kind kind = Kind::Table;
    std::uint32_t variant = 0;
    std::uint32_t stride = 16;
    std::uint32_t vertices = 3;
    std::uint32_t texture = 0;
};

bool RingKind(Kind kind) {
    return kind == Kind::Ring || kind == Kind::Inner || kind == Kind::RingInner || kind == Kind::Sized || kind == Kind::Format;
}

std::size_t OuterSlot(const State& state) {
    switch (state.kind) {
    case Kind::Ring: return state.variant % 6;
    case Kind::RingInner: return 6 + state.variant % 6;
    case Kind::Inner: return 12;
    case Kind::Sized: return 13;
    default: return 14;
    }
}

std::size_t InnerAt(const State& state) {
    switch (state.kind) {
    case Kind::Inner:
    case Kind::RingInner: return InnerOffset + (state.variant % 6) * InnerBytes;
    case Kind::Sized: return InnerOffset + 6 * InnerBytes;
    default: return InnerOffset + 7 * InnerBytes;
    }
}

void Apply(const State& state) {
    auto& blocks = *Scene;
    if (RingKind(state.kind)) {
        const auto records = state.kind == Kind::Sized ? InnerRecords * (1 + state.variant % 6) : InnerRecords;
        const auto format = state.kind == Kind::Format ? RawFormats[state.variant % RawFormats.size()] : RawFormats[0];
        WriteWords(blocks.ring, RingSlotsOffset + OuterSlot(state) * RingSlotBytes + 16u, RawDescriptor(blocks.ring.At(InnerAt(state)), records, format));
        return;
    }
    switch (state.kind) {
    case Kind::Constants: {
        const float value = state.variant == 0 ? 0.25f : 0.5f;
        std::memcpy(blocks.data.Bytes(ConstantsOffset + state.variant * 0x40u), &value, sizeof(value));
        break;
    }
    case Kind::Table:
        WriteWords(blocks.vertices, TableOffset, VertexDescriptor(blocks.vertices.At(state.variant == 0 ? NearOffset : FarOffset)));
        WriteWords(blocks.vertices, TableOffset + 16u, VertexDescriptor(blocks.vertices.At(ShiftOffset), state.stride));
        break;
    case Kind::Fetch:
        WriteWords(blocks.fetch, FetchBufferTableOffset, SlotDescriptor(blocks.fetch.At(state.variant == 0 ? CurrentSlotsOffset : ShiftedSlotsOffset)));
        break;
    case Kind::FetchFormat:
        WriteWords(blocks.fetch, FetchBufferTableOffset, SlotDescriptor(blocks.fetch.At(CurrentSlotsOffset), SlotFormats[state.variant % SlotFormats.size()]));
        break;
    case Kind::Texture: {
        std::array<std::uint8_t, TextureBytes> wanted{};
        for (std::size_t at = 0; at < wanted.size(); at += Texels[0].size()) std::memcpy(wanted.data() + at, Texels[state.variant % Texels.size()].data(), Texels[0].size());
        auto* texels = blocks.texture->Bytes(TextureOffsets[state.texture % TextureOffsets.size()]);
        if (std::memcmp(texels, wanted.data(), wanted.size()) != 0) std::memcpy(texels, wanted.data(), wanted.size());
        break;
    }
    default:
        break;
    }
}

void Submit(const State& state, std::uint32_t draws) {
    auto& blocks = *Scene;
    switch (state.kind) {
    case Kind::Constants: {
        const auto pointer = blocks.data.At(ConstantsOffset + state.variant * 0x40u);
        SubmitDraws(reinterpret_cast<std::uintptr_t>(PlainVertex.data()), reinterpret_cast<std::uintptr_t>(ConstantPixel.data()), {{0x242, 4}}, {{0x8b, 0}, {0x8c, 0}, {0x00b, 3u << 1u}, {0x00c, static_cast<std::uint32_t>(pointer)}, {0x00d, static_cast<std::uint32_t>(pointer >> 32u)}, {0x00e, std::bit_cast<std::uint32_t>(0.25f)}}, blocks.target, draws, state.vertices);
        break;
    }
    case Kind::Table: {
        const auto table = blocks.vertices.At(TableOffset);
        const auto pointer = blocks.data.At(0);
        SubmitDraws(reinterpret_cast<std::uintptr_t>(TableVertex.data()), reinterpret_cast<std::uintptr_t>(ReadPixel.data()), {{0x242, 4}}, {{0x8b, 2u << 1u}, {0x8c, static_cast<std::uint32_t>(table)}, {0x8d, static_cast<std::uint32_t>(table >> 32u)}, {0x00b, 8u << 1u}, {0x00c, 0}, {0x00d, 0}, {0x00e, static_cast<std::uint32_t>(pointer)}, {0x00f, static_cast<std::uint32_t>(pointer >> 32u)}}, blocks.target, draws);
        break;
    }
    case Kind::Fetch:
    case Kind::FetchFormat: {
        const auto stored = OutputDescriptor(blocks.output.At(0));
        const auto buffers = blocks.fetch.At(FetchBufferTableOffset);
        const auto attributes = blocks.fetch.At(FetchAttribTableOffset);
        const std::array<std::uint32_t, 10> user{stored[0], stored[1], stored[2], stored[3], static_cast<std::uint32_t>(buffers), static_cast<std::uint32_t>(buffers >> 32u), static_cast<std::uint32_t>(attributes), static_cast<std::uint32_t>(attributes >> 32u), FetchVertexOffset, 0u};
        const auto pointer = blocks.data.At(0);
        RegisterWrites registers{{0x8b, static_cast<std::uint32_t>(user.size()) << 1u}};
        for (std::uint32_t word = 0; word < user.size(); ++word) registers.emplace_back(0x8cu + word, user[word]);
        for (const auto& entry : RegisterWrites{{0x00b, 8u << 1u}, {0x00c, 0}, {0x00d, 0}, {0x00e, static_cast<std::uint32_t>(pointer)}, {0x00f, static_cast<std::uint32_t>(pointer >> 32u)}}) registers.push_back(entry);
        SubmitDraws(reinterpret_cast<std::uintptr_t>(FetchVertex.data()), reinterpret_cast<std::uintptr_t>(ReadPixel.data()), {{0x242, 4}, {0x24a, FetchIndexOffset}}, registers, blocks.target, draws, state.vertices);
        break;
    }
    case Kind::Texture: {
        const auto image = TextureDescriptor(blocks.texture->At(TextureOffsets[state.texture % TextureOffsets.size()]));
        RegisterWrites registers{{0x8b, 0}, {0x8c, 0}, {0x00b, TextureUserWords << 1u}};
        for (std::uint32_t word = 0; word < TextureUserWords; ++word) registers.emplace_back(0x00cu + word, word < image.size() ? image[word] : 0u);
        SubmitDraws(reinterpret_cast<std::uintptr_t>(PlainVertex.data()), reinterpret_cast<std::uintptr_t>(SamplePixel.data()), {{0x242, 4}, {0x24a, 0}}, registers, blocks.target, draws, state.vertices);
        break;
    }
    default: {
        const auto constants = RawDescriptor(blocks.ring.At(RingSlotsOffset + OuterSlot(state) * RingSlotBytes), ConstantRecords);
        SubmitDraws(reinterpret_cast<std::uintptr_t>(PlainVertex.data()), reinterpret_cast<std::uintptr_t>(RingPixel.data()), {{0x242, 4}}, {{0x8b, 0}, {0x8c, 0}, {0x00b, 4u << 1u}, {0x00c, constants[0]}, {0x00d, constants[1]}, {0x00e, constants[2]}, {0x00f, constants[3]}}, blocks.target, draws, state.vertices);
        break;
    }
    }
}

std::vector<std::uint8_t> Output() {
    auto& output = Scene->output;
    AgcDriver::GuestMemory::FlushGpuWrites(static_cast<std::uintptr_t>(output.At(0)), OutputSlots * 16u);
    return {output.Bytes(0), output.Bytes(0) + OutputSlots * 16u};
}

std::vector<std::uint8_t> Render(const State& state, std::uint32_t draws) {
    auto& blocks = *Scene;
    std::memset(blocks.target.Bytes(0), 0, ColorBytes);
    std::vector<std::uint32_t> unwritten(OutputSlots * 4u, Unwritten);
    std::memcpy(blocks.output.Bytes(0), unwritten.data(), unwritten.size() * sizeof(std::uint32_t));
    Apply(state);
    Submit(state, draws);
    auto image = Image(blocks.target);
    if (state.kind == Kind::Fetch || state.kind == Kind::FetchFormat) {
        const auto written = Output();
        image.insert(image.end(), written.begin(), written.end());
    }
    return image;
}

struct Totals {
    std::array<std::uint64_t, static_cast<std::size_t>(Counter::Count)> counters{};
    std::uint64_t differences = 0;
    std::uint64_t fetches = 0;

    static Totals Now() {
        Totals totals;
        for (std::size_t index = 0; index < totals.counters.size(); ++index) totals.counters[index] = DrawRecords::Total(static_cast<Counter>(index));
        totals.differences = DrawRecords::Differences();
        totals.fetches = DrawRecords::Checked(DrawRecords::Item::Fetch);
        return totals;
    }

    std::uint64_t Since(const Totals& base, Counter counter) const { return counters[static_cast<std::size_t>(counter)] - base.counters[static_cast<std::size_t>(counter)]; }
};

std::string Describe(const Totals& now, const Totals& base) {
    std::string text;
    for (std::size_t index = 0; index < now.counters.size(); ++index) {
        const auto moved = now.Since(base, static_cast<Counter>(index));
        if (moved != 0) text += " counter" + std::to_string(index) + "=" + std::to_string(moved);
    }
    return text;
}

namespace RuntimeSwitches = AgcDriver::Graphics::RuntimeSwitches;
using AgcDriver::Graphics::RuntimeSwitch;

struct Arms {
    bool ring = false;

    template <typename Step>
    void Run(const std::string& name, const std::vector<State>& states, std::uint32_t draws, const Step& check) {
        std::vector<std::vector<std::uint8_t>> today;
        RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, false);
        RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, true);
        RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, false);
        for (const auto& state : states) today.push_back(Render(state, draws));
        RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, true);
        const auto base = Totals::Now();
        for (std::size_t index = 0; index < states.size(); ++index) {
            const auto image = Render(states[index], draws);
            Require(image == today[index], name + ": submission " + std::to_string(index) + " with draw records rendered differently from today's path" + Describe(Totals::Now(), base));
        }
        const auto now = Totals::Now();
        std::cout << name << ":" << Describe(now, base) << '\n';
        if (ring) check(now, base);
        RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, false);
    }
};

std::uint64_t RecordPathHits(const Totals& now, const Totals& base) {
    return now.Since(base, Counter::Hits) - now.Since(base, Counter::HitsOnTodaysPath);
}

void ExpectRecordPath(const std::string& name, const Totals& now, const Totals& base) {
    Require(now.differences == base.differences, name + ": the draw record verify found differences" + Describe(now, base));
    Require(now.Since(base, Counter::PlansBuilt) >= 1 && now.Since(base, Counter::Hits) >= DrawsPerSubmission, name + ": no fetch plan was built, or its identity did not hit" + Describe(now, base));
    Require(DrawRecords::VerifyEvery() == 1 || RecordPathHits(now, base) != 0, name + ": no hit skipped today's path" + Describe(now, base));
    Require(DrawRecords::VerifyEvery() == 0 || now.fetches > base.fetches, name + ": the verify checked no hit" + Describe(now, base));
}

bool CommitsRun() {
    return DrawRecords::VerifyEvery() != 1;
}

void ExpectCommits(const std::string& name, const Totals& now, const Totals& base, bool moves) {
    if (!CommitsRun()) {
        Require(now.Since(base, Counter::CommitHits) == 0, name + ": the committer recorded a hit the finisher verified" + Describe(now, base));
        return;
    }
    const auto recorded = now.Since(base, Counter::CommitHits);
    const auto held = now.Since(base, Counter::CommitPassHeld);
    Require(now.Since(base, Counter::CommitBuilt) >= 1, name + ": no committer half was built from today's draw" + Describe(now, base));
    Require(recorded != 0, name + ": no record hit was recorded by the committer half" + Describe(now, base));
    Require(held < recorded, name + ": the committer held targets across a pass break" + Describe(now, base));
    if (DrawRecords::VerifyEvery() == 0) {
        Require(held != 0, name + ": the committer did not hold targets within a pass" + Describe(now, base));
        Require(now.Since(base, Counter::CommitKeptOnce) != 0 && now.Since(base, Counter::CommitNotesSkipped) != 0, name + ": record hits did not keep once per batch and note their targets once per pass" + Describe(now, base));
    }
    if (moves) Require(now.Since(base, Counter::CommitPushed) != 0 || now.Since(base, Counter::CommitUnpushed) != 0, name + ": record hits with moved buffers were neither pushed nor counted without a push" + Describe(now, base));
    if (DrawRecords::VerifyEvery() > 1) Require(DrawRecords::Checked(DrawRecords::Item::Commit) != 0, name + ": the committer verify checked no hit" + Describe(now, base));
}

std::vector<State> Alternating(Kind kind, std::size_t count) {
    std::vector<State> states;
    for (std::size_t index = 0; index < count; ++index) states.push_back({kind, static_cast<std::uint32_t>(index & 1u), 16});
    return states;
}

std::atomic<bool> HookArmed{false};
std::atomic<int> HookChanges{0};

void MoveTableBeforeCheck() {
    if (!HookArmed.exchange(false)) return;
    WriteWords(Scene->vertices, TableOffset, VertexDescriptor(Scene->vertices.At(FarOffset)));
    HookChanges.fetch_add(1);
}

void Prepare() {
    auto& blocks = *Scene;
    WriteVertices(blocks.vertices, NearOffset, -8.0f);
    WriteVertices(blocks.vertices, ShiftOffset, 8.0f);
    WriteVertices(blocks.vertices, StillOffset, 0.0f);
    WriteVertices(blocks.vertices, FarOffset, -4.0f);
    WriteWords(blocks.vertices, TableOffset + 32u, VertexDescriptor(blocks.vertices.At(StillOffset)));
    const float one = 1.0f;
    std::memcpy(blocks.data.Bytes(0), &one, sizeof(one));
    const auto slots = [&](std::size_t offset, std::uint32_t shift) {
        for (std::uint32_t record = 0; record < SlotRecords; ++record) {
            const std::uint32_t slot = record + shift;
            std::memcpy(blocks.fetch.Bytes(offset + record * sizeof(slot)), &slot, sizeof(slot));
        }
    };
    slots(ShiftedSlotsOffset, SlotShift);
    slots(CurrentSlotsOffset, 0);
    std::memset(blocks.fetch.Bytes(InstanceSlotsOffset), 0, SlotRecords * sizeof(std::uint32_t));
    const std::array<std::uint32_t, 2> attributes{0u, 1u | (1u << 26u)};
    std::memcpy(blocks.fetch.Bytes(FetchAttribTableOffset), attributes.data(), sizeof(attributes));
    WriteWords(blocks.fetch, FetchBufferTableOffset + 16u, SlotDescriptor(blocks.fetch.At(InstanceSlotsOffset)));
    for (std::size_t slot = 0; slot < RingSlots; ++slot) {
        const float red = 0.125f * static_cast<float>(1 + slot % 7);
        std::memcpy(blocks.ring.Bytes(RingSlotsOffset + slot * RingSlotBytes), &red, sizeof(red));
    }
    for (std::size_t inner = 0; inner < InnerBuffers; ++inner) {
        const float green = 0.1f + 0.1f * static_cast<float>(inner);
        std::memcpy(blocks.ring.Bytes(InnerOffset + inner * InnerBytes), &green, sizeof(green));
    }
}

void Hits(Arms& arms) {
    arms.Run("constants behind a moving user data pointer", Alternating(Kind::Constants, 6), DrawsPerSubmission, [](const Totals& now, const Totals& base) {
        ExpectRecordPath("constants", now, base);
        Require(now.Since(base, Counter::MissContentDiffers) == 0, "constants: draws whose flattened constants changed missed on content" + Describe(now, base));
        ExpectCommits("constants", now, base, true);
        Require(now.Since(base, Counter::CommitUnpushed) != 0 || now.Since(base, Counter::CommitPushed) != 0 || !CommitsRun(), "constants: no record hit bound its moved constants through push descriptors" + Describe(now, base));
    });
    arms.Run("a vertex buffer descriptor moving inside a memory table", Alternating(Kind::Table, 6), DrawsPerSubmission, [](const Totals& now, const Totals& base) {
        ExpectRecordPath("table", now, base);
        Require(now.Since(base, Counter::MissContentDiffers) == 0, "table: draws whose read-only buffer descriptor moved inside its table missed on content" + Describe(now, base));
    });
    arms.Run("a fetched vertex buffer moving", Alternating(Kind::Fetch, 6), DrawsPerSubmission, [](const Totals& now, const Totals& base) {
        ExpectRecordPath("fetch", now, base);
        Require(now.Since(base, Counter::InputRefetches) != 0 && now.Since(base, Counter::MissVertexLayout) == 0, "fetch: hits whose fetched vertex buffer moved did not refetch their inputs" + Describe(now, base));
    });
}

std::vector<State> Moving(Kind kind, std::size_t count) {
    std::vector<State> states;
    for (std::size_t index = 0; index < count; ++index) states.push_back({kind, static_cast<std::uint32_t>(index), 16});
    return states;
}

void Ring(Arms& arms) {
    arms.Run("an inline constant buffer descriptor read through and bound, moving to a new ring slot every submission", Moving(Kind::Ring, 6), DrawsPerSubmission, [](const Totals& now, const Totals& base) {
        ExpectRecordPath("ring", now, base);
        Require(now.Since(base, Counter::MissUserWordBaseValue) == 0 && now.Since(base, Counter::MissContentDiffers) == 0, "ring: draws whose inline buffer descriptor moved with the runs read through it missed" + Describe(now, base));
        Require(now.Since(base, Counter::HitsBaseValue) != 0, "ring: no hit relocated a base and value user word" + Describe(now, base));
        Require(now.Since(base, Counter::RecordsBuilt) == 1, "ring: the moving descriptor built more than one record" + Describe(now, base));
    });
    arms.Run("a read-only buffer descriptor moving inside a fixed constant buffer", Moving(Kind::Inner, 6), DrawsPerSubmission, [](const Totals& now, const Totals& base) {
        ExpectRecordPath("inner", now, base);
        Require(now.Since(base, Counter::MissContentDiffers) == 0 && now.Since(base, Counter::MissBufferFields) == 0, "inner: draws whose read-only buffer base moved inside a captured run missed" + Describe(now, base));
        Require(now.Since(base, Counter::HitsBuffersPatched) != 0, "inner: no hit patched a moved buffer base" + Describe(now, base));
    });
    arms.Run("a read-only buffer descriptor moving inside an inline constant buffer that moves too", Moving(Kind::RingInner, 6), DrawsPerSubmission, [](const Totals& now, const Totals& base) {
        ExpectRecordPath("ring and inner", now, base);
        Require(now.Since(base, Counter::MissUserWordBaseValue) == 0 && now.Since(base, Counter::MissContentDiffers) == 0, "ring and inner: draws whose descriptors moved missed" + Describe(now, base));
        Require(now.Since(base, Counter::HitsBaseValue) != 0 && now.Since(base, Counter::HitsBuffersPatched) != 0, "ring and inner: hits did not both relocate the inline descriptor and patch the moved buffer base" + Describe(now, base));
    });
    arms.Run("a read-only buffer descriptor whose record count changes", Moving(Kind::Sized, 6), DrawsPerSubmission, [](const Totals& now, const Totals& base) {
        ExpectRecordPath("sized", now, base);
        Require(now.Since(base, Counter::MissContentDiffers) == 0 && now.Since(base, Counter::HitsBuffersPatched) != 0, "sized: draws whose read-only buffer size changed inside a captured run did not hit with the patched size" + Describe(now, base));
    });
}

void Formats(Arms& arms) {
    const auto passes = 3u;
    const auto formats = static_cast<std::uint32_t>(RawFormats.size());
    std::vector<State> states;
    for (std::uint32_t pass = 0; pass < passes; ++pass) {
        for (std::uint32_t format = 0; format < formats; ++format) states.push_back({Kind::Format, format, 16});
    }
    std::vector<std::vector<std::uint8_t>> today;
    RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, false);
    RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, true);
    RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, false);
    for (const auto& state : states) today.push_back(Render(state, DrawsPerSubmission));
    RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, true);
    const auto base = Totals::Now();
    auto warmed = base;
    for (std::size_t index = 0; index < states.size(); ++index) {
        Require(Render(states[index], DrawsPerSubmission) == today[index], "formats: submission " + std::to_string(index) + " with draw records rendered differently from today's path" + Describe(Totals::Now(), base));
        if (index + 1 == formats) warmed = Totals::Now();
    }
    const auto now = Totals::Now();
    std::cout << "a descriptor format word cycling through " << formats << " values:" << Describe(now, base) << '\n';
    if (arms.ring) {
        Require(now.differences == base.differences, "formats: the draw record verify found differences" + Describe(now, base));
        Require(warmed.Since(base, Counter::MissContentDiffers) != 0, "formats: a changed descriptor format word did not miss on content" + Describe(warmed, base));
        Require(now.Since(warmed, Counter::MissContentDiffers) == 0, "formats: once every format had a record, a draw missed on content, so the variant pre-check did not pick the record of the draw's format" + Describe(now, warmed));
        Require(now.Since(warmed, Counter::Hits) >= DrawsPerSubmission, "formats: draws of formats that had records did not hit" + Describe(now, warmed));
    }
    RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, false);
}

void Misses(Arms& arms) {
    auto states = Alternating(Kind::Table, 4);
    states.push_back({Kind::Table, 0, 32});
    states.push_back({Kind::Table, 1, 32});
    arms.Run("a structural descriptor word changing", states, DrawsPerSubmission, [](const Totals& now, const Totals& base) {
        Require(now.differences == base.differences, "structural: the draw record verify found differences" + Describe(now, base));
        Require(now.Since(base, Counter::MissContentDiffers) != 0, "structural: draws whose buffer stride changed in memory did not miss on content" + Describe(now, base));
    });
    AgcDriver::DriverDetail::Driver::SetCaptureCheckHook(&MoveTableBeforeCheck);
    RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, false);
    RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, true);
    RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, false);
    const auto nearImage = Render({Kind::Table, 0, 16}, 1);
    const auto farImage = Render({Kind::Table, 1, 16}, 1);
    RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, true);
    const auto base = Totals::Now();
    for (int round = 0; round < 4; ++round) Require(Render({Kind::Table, static_cast<std::uint32_t>(round & 1), 16}, 1) == (round & 1 ? farImage : nearImage), "changed: a warm-up draw rendered differently from today's path");
    std::uint64_t changed = 0;
    for (int attempt = 0; attempt < 8 && changed == 0; ++attempt) {
        auto& blocks = *Scene;
        std::memset(blocks.target.Bytes(0), 0, ColorBytes);
        Apply({Kind::Table, 0, 16});
        HookArmed.store(true);
        Submit({Kind::Table, 0, 16}, 1);
        HookArmed.store(false);
        Require(Image(blocks.target) == farImage, "changed: a draw whose buffer descriptor moved after the fetch rendered the descriptor fetched before the move");
        changed = Totals::Now().Since(base, Counter::MissMemoryChanged);
    }
    AgcDriver::DriverDetail::Driver::SetCaptureCheckHook(nullptr);
    const auto now = Totals::Now();
    std::cout << "a descriptor moved between the fetch and the commit:" << Describe(now, base) << '\n';
    if (arms.ring) Require(changed != 0 && now.differences == base.differences, "changed: a CPU write after the record's fetch did not send the draw to today's path" + Describe(now, base));
    RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, false);
}

void Fallbacks(Arms& arms) {
    using AgcDriver::Graphics::DrawRecipeMiss;
    using AgcDriver::Graphics::FailRecordChecks;
    const std::array<std::pair<DrawRecipeMiss, Counter>, 4> checks{{
        {DrawRecipeMiss::Proof, Counter::CommitProof},
        {DrawRecipeMiss::TargetGone, Counter::CommitTargets},
        {DrawRecipeMiss::Moved, Counter::CommitMoved},
        {DrawRecipeMiss::ObjectsGone, Counter::CommitObjects},
    }};
    for (const auto& [check, counter] : checks) {
        const auto name = std::string("a failed committer check (") + AgcDriver::Graphics::DrawRecipeMissName(check) + ")";
        FailRecordChecks(check, 2);
        arms.Run(name, Alternating(Kind::Constants, 10), DrawsPerSubmission, [&, counter = counter](const Totals& now, const Totals& base) {
            Require(now.differences == base.differences, name + ": the draw record verify found differences" + Describe(now, base));
            if (!CommitsRun()) return;
            Require(now.Since(base, counter) != 0, name + ": the failed check did not send the draw to today's path" + Describe(now, base));
            Require(now.Since(base, Counter::CommitBuilt) >= 2 && now.Since(base, Counter::CommitHits) != 0, name + ": today's path did not rebuild the committer half, or the record did not hit again after it" + Describe(now, base));
        });
        FailRecordChecks(check, 0);
    }
}

void Injected(Arms& arms) {
    if (DrawRecords::VerifyEvery() == 0) return;
    const auto before = DrawRecords::Differences();
    DrawRecords::CorruptAnswers(true);
    arms.Run("a corrupted record answer", Alternating(Kind::Constants, 6), DrawsPerSubmission, [&](const Totals& now, const Totals& base) {
        Require(now.differences > base.differences && now.Since(base, Counter::Demotions) != 0, "inject: the verify did not catch a corrupted record answer" + Describe(now, base));
        Require(DrawRecords::Differences(DrawRecords::Field::Parameters) != 0, "inject: the verify caught the corrupted answer in another field");
    });
    DrawRecords::CorruptAnswers(false);
    if (arms.ring) Require(DrawRecords::Differences() > before, "inject: no difference was counted");
}

namespace StagePlanReport = AgcDriver::DriverDetail::StagePlans;
using PlanCounter = StagePlanReport::Counter;

struct PlanTotals {
    std::array<std::uint64_t, static_cast<std::size_t>(PlanCounter::Count)> counters{};
    std::uint64_t differences = 0;
    std::uint64_t checked = 0;
    std::uint64_t commits = 0;

    static PlanTotals Now() {
        PlanTotals totals;
        for (std::size_t index = 0; index < totals.counters.size(); ++index) totals.counters[index] = StagePlanReport::Total(static_cast<PlanCounter>(index));
        totals.differences = StagePlanReport::Differences() + DrawRecords::Differences();
        totals.checked = StagePlanReport::Checked(DrawRecords::Item::Fetch);
        totals.commits = DrawRecords::Total(Counter::CommitHits);
        return totals;
    }

    std::uint64_t Since(const PlanTotals& base, PlanCounter counter) const { return counters[static_cast<std::size_t>(counter)] - base.counters[static_cast<std::size_t>(counter)]; }
    std::uint64_t Since(const PlanTotals& base, PlanCounter first, PlanCounter last) const {
        std::uint64_t total = 0;
        for (auto index = static_cast<std::size_t>(first); index <= static_cast<std::size_t>(last); ++index) total += Since(base, static_cast<PlanCounter>(index));
        return total;
    }
};

std::string DescribePlans(const PlanTotals& now, const PlanTotals& base) {
    std::string text;
    for (std::size_t index = 0; index < now.counters.size(); ++index) {
        const auto moved = now.Since(base, static_cast<PlanCounter>(index));
        if (moved != 0) text += std::string("; ") + StagePlanReport::CounterName(static_cast<PlanCounter>(index)) + " " + std::to_string(moved);
    }
    if (now.commits != base.commits) text += "; committer hits " + std::to_string(now.commits - base.commits);
    return text;
}

struct PlanArms {
    bool ring = false;

    template <typename Step>
    void Run(const std::string& name, const std::vector<State>& states, std::uint32_t draws, const Step& check, bool clean = true) {
        std::vector<std::vector<std::uint8_t>> today;
        RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, true);
        RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, false);
        RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, false);
        for (const auto& state : states) today.push_back(Render(state, draws));
        RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, true);
        const auto base = PlanTotals::Now();
        for (std::size_t index = 0; index < states.size(); ++index) {
            const auto image = Render(states[index], draws);
            Require(image == today[index], name + ": submission " + std::to_string(index) + " with stage plans rendered differently from today's path" + DescribePlans(PlanTotals::Now(), base));
        }
        const auto now = PlanTotals::Now();
        std::cout << "stage plans, " << name << ":" << DescribePlans(now, base) << '\n';
        if (ring) {
            Require(!clean || now.differences == base.differences, name + ": the stage plan or committer verify found differences" + DescribePlans(now, base));
            check(now, base);
        }
        RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, false);
    }
};

std::uint64_t NewKeyMisses(const PlanTotals& now, const PlanTotals& base) {
    return now.Since(base, PlanCounter::MissNewLayout, PlanCounter::MissNewCombination);
}

void ExpectPlanHits(const std::string& name, const PlanTotals& now, const PlanTotals& base, std::uint64_t at) {
    const auto hits = now.Since(base, PlanCounter::Hits);
    Require(hits >= at, name + ": " + std::to_string(hits) + " draws hit stage plans, expected at least " + std::to_string(at) + DescribePlans(now, base));
    if (StagePlanReport::VerifyEvery() == 1) Require(now.Since(base, PlanCounter::HitsOnTodaysPath) == hits, name + ": a hit skipped the verify of every hit" + DescribePlans(now, base));
    else Require(now.Since(base, PlanCounter::HitsOnTodaysPath) < hits, name + ": no stage plan hit skipped today's path" + DescribePlans(now, base));
    if (StagePlanReport::VerifyEvery() != 0) Require(now.checked > base.checked, name + ": the stage plan verify checked no hit" + DescribePlans(now, base));
    if (StagePlanReport::VerifyEvery() != 1 && DrawRecords::VerifyEvery() != 1) Require(now.commits > base.commits, name + ": the committer half recorded no stage plan hit" + DescribePlans(now, base));
}

std::vector<State> MovingIdentities(Kind kind, std::size_t count) {
    std::vector<State> states;
    for (std::size_t index = 0; index < count; ++index) states.push_back({kind, static_cast<std::uint32_t>(index), 16, (index & 1u) != 0 ? 6u : 3u});
    return states;
}

void Plans(PlanArms& arms) {
    arms.Run("draws of other identities whose constant buffer and the table in it move every submission", MovingIdentities(Kind::RingInner, 6), DrawsPerSubmission, [](const PlanTotals& now, const PlanTotals& base) {
        ExpectPlanHits("moved", now, base, DrawsPerSubmission);
        Require(now.Since(base, PlanCounter::DrawPlansAdded) == 1 && now.Since(base, PlanCounter::StagePlansAdded) == 2, "moved: draws whose descriptors moved did not share one draw plan of two stage plans" + DescribePlans(now, base));
        Require(NewKeyMisses(now, base) == 0 && now.Since(base, PlanCounter::MissContentDiffers) == 0 && now.Since(base, PlanCounter::MissBufferFields) == 0, "moved: a draw whose constant buffer and table moved missed its stage plan" + DescribePlans(now, base));
        Require(now.Since(base, PlanCounter::HitsRelocated) != 0 && now.Since(base, PlanCounter::HitsBaseValue) != 0 && now.Since(base, PlanCounter::HitsBuffersPatched) != 0, "moved: hits did not relocate the moved constant buffer's runs, patch its inline descriptor and patch the buffer moved inside it" + DescribePlans(now, base));
    });
    const auto formats = static_cast<std::uint32_t>(RawFormats.size());
    std::vector<State> cycled;
    for (std::uint32_t pass = 0; pass < 3; ++pass) {
        for (std::uint32_t format = 0; format < formats; ++format) cycled.push_back({Kind::Format, format, 16, pass == 1 ? 6u : 3u});
    }
    arms.Run("a buffer format word in the key cycling through " + std::to_string(formats) + " values", cycled, DrawsPerSubmission, [&](const PlanTotals& now, const PlanTotals& base) {
        ExpectPlanHits("formats", now, base, DrawsPerSubmission * 2);
        Require(now.Since(base, PlanCounter::MissNewRunWords) != 0, "formats: a changed format word did not miss on a new key" + DescribePlans(now, base));
        Require(now.Since(base, PlanCounter::DrawPlansAdded) == formats && now.Since(base, PlanCounter::StagePlansAdded) == formats + 1 && now.Since(base, PlanCounter::StagePlansReplaced) == 0, "formats: each format did not get its own pixel stage plan and draw plan beside one vertex stage plan" + DescribePlans(now, base));
        Require(now.Since(base, PlanCounter::Builds) <= DrawsPerSubmission * formats, "formats: draws of formats that had stage plans kept building" + DescribePlans(now, base));
    });
    std::vector<State> fetched;
    for (std::uint32_t index = 0; index < 6; ++index) fetched.push_back({Kind::FetchFormat, (index / 2u) % 2u, 16, (index & 1u) != 0 ? 6u : 3u});
    arms.Run("a vertex buffer format, outside the key, that the vertex stage's module depends on", fetched, DrawsPerSubmission, [](const PlanTotals& now, const PlanTotals& base) {
        Require(now.Since(base, PlanCounter::MissVertexLayout) != 0, "vertex format: a vertex buffer format change under one stage plan key did not fall back on its vertex layout" + DescribePlans(now, base));
        Require(now.Since(base, PlanCounter::StagePlansReplaced) != 0 && now.Since(base, PlanCounter::Hits) != 0, "vertex format: the fallback did not replace the stage plan with one for the new format, or no draw hit" + DescribePlans(now, base));
    });
}

std::uint64_t ProofsRun(const PlanTotals& now, const PlanTotals& base) {
    return now.Since(base, PlanCounter::ProofFirst, PlanCounter::ProofUnsettled) + now.Since(base, PlanCounter::ProofsFull);
}

std::uint64_t ProofsSkipped(const PlanTotals& now, const PlanTotals& base) {
    return now.Since(base, PlanCounter::ProofsSkipped) + now.Since(base, PlanCounter::ProofsRequeried);
}

void Textures(PlanArms& arms) {
    auto& blocks = *Scene;
    if (blocks.texture == nullptr) blocks.texture = std::make_unique<TextureBlock>();
    const auto counts = TextureVertexCounts.size();
    RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, true);
    RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, false);
    RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, false);
    std::array<std::array<std::vector<std::uint8_t>, TextureVertexCounts.size()>, Texels.size()> today;
    for (std::uint32_t texel = 0; texel < Texels.size(); ++texel) {
        for (std::size_t count = 0; count < counts; ++count) today[texel][count] = Render({Kind::Texture, texel, 16, TextureVertexCounts[count], 0}, TextureDraws);
    }
    Require(today[0][0] != today[1][0], "textures: the two texel values rendered the same image");
    Apply({Kind::Texture, 0, 16, 3, 1});
    const bool armed = blocks.texture->Arm(TextureOffsets[1]);
#ifdef _WIN32
    Require(armed, "textures: two collects of the whole resident texture block did not arm it");
#endif
    RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, true);
    const auto base = PlanTotals::Now();
    for (std::size_t index = 0; index < 2 * counts; ++index) {
        const auto count = index % counts;
        Require(Render({Kind::Texture, 0, 16, TextureVertexCounts[count], 1}, TextureDraws) == today[0][count], "textures: submission " + std::to_string(index) + " of planned draws sampling a clean texture rendered differently from today's path" + DescribePlans(PlanTotals::Now(), base));
    }
    const auto clean = PlanTotals::Now();
    std::cout << "stage plans, draws of " << counts << " identities sampling a clean texture in armed memory:" << DescribePlans(clean, base) << '\n';
    Require(!armed || blocks.texture->Armed(TextureOffsets[1]), "textures: the clean texture's memory was disarmed without a write");
    Apply({Kind::Texture, 1, 16, 3, 1});
    Require(!blocks.texture->Armed(TextureOffsets[1]), "textures: a CPU write into armed texture memory left it armed");
    for (std::size_t count = 0; count < 2; ++count) {
        Require(Render({Kind::Texture, 1, 16, TextureVertexCounts[count], 1}, TextureDraws) == today[1][count], "textures: planned draws after a CPU write into their texture rendered differently from today's path" + DescribePlans(PlanTotals::Now(), clean));
    }
    const auto written = PlanTotals::Now();
    std::cout << "stage plans, the texture written by the CPU between two submissions:" << DescribePlans(written, clean) << '\n';
    RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, false);
    if (!arms.ring) return;
    Require(written.differences == base.differences, "textures: the stage plan or committer verify found differences" + DescribePlans(written, base));
    ExpectPlanHits("textures", clean, base, DrawsPerSubmission);
    Require(clean.Since(base, PlanCounter::DrawPlansAdded) == 1, "textures: draws of " + std::to_string(counts) + " identities did not share one draw plan" + DescribePlans(clean, base));
    if (StagePlanReport::VerifyEvery() == 1 || DrawRecords::VerifyEvery() == 1) return;
    Require(clean.Since(base, PlanCounter::TargetsHeld, PlanCounter::TargetsChanged) != 0 && ProofsRun(clean, base) != 0, "textures: the committer half of planned draws neither held its targets nor proved its template" + DescribePlans(clean, base));
    if (RuntimeSwitches::Set(RuntimeSwitch::NoPlanCleanProofs)) {
        Require(ProofsSkipped(written, base) == 0 && written.Since(base, PlanCounter::ProofsFull) != 0, "textures: with clean proofs off, a planned draw skipped its template proof or none was proved in full" + DescribePlans(written, base));
    } else {
        Require(!armed || ProofsSkipped(clean, base) != 0, "textures: no planned draw of a clean texture in armed memory skipped its template proof" + DescribePlans(clean, base));
        Require(ProofsRun(written, clean) != 0, "textures: no planned draw proved its template again after a CPU write into its texture" + DescribePlans(written, clean));
    }
    if (!AgcDriver::ThreadReport::Enabled()) {
        AgcDriver::ThreadReport::MarkItem(AgcDriver::ThreadReport::ItemMark::Planned);
        Require(AgcDriver::ThreadReport::TakeItemMark() == AgcDriver::ThreadReport::ItemMark::None, "textures: a planned item was marked while the thread report is off");
        return;
    }
    using ThreadCount = AgcDriver::ThreadReport::Count;
    Require(AgcDriver::ThreadReport::Peek(ThreadCount::CommitterPlannedDraws) != 0 && AgcDriver::ThreadReport::Peek(ThreadCount::FinisherPlannedJobs) != 0 && AgcDriver::ThreadReport::Peek(ThreadCount::FinisherPlanMisses) != 0, "textures: the thread report counted no planned committer draw, planned finisher job or plan miss");
    AgcDriver::ThreadReport::Phase();
}

void InjectedPlans(PlanArms& arms) {
    Require(StagePlanReport::VerifyEvery() == 1, "inject plans: the stage plan verify needs APS5_VERIFY_STAGE_PLANS=1");
    DrawRecords::CorruptAnswers(true);
    arms.Run("a corrupted stage plan answer", MovingIdentities(Kind::Constants, 6), DrawsPerSubmission, [](const PlanTotals&, const PlanTotals&) {}, false);
    DrawRecords::CorruptAnswers(false);
    if (!arms.ring) return;
    Require(StagePlanReport::Differences(DrawRecords::Field::Parameters) != 0 && StagePlanReport::Total(PlanCounter::Demotions) != 0 && StagePlanReport::Total(PlanCounter::MissRefused) != 0, "inject plans: the verify did not catch a corrupted stage plan answer, or the shape was not refused after it");
}

void InjectedCommit(Arms& arms) {
    Require(DrawRecords::VerifyEvery() >= 2, "inject commit: the committer verify needs APS5_VERIFY_DRAW_RECORDS of 2 or more, so hits reach the committer");
    DrawRecords::CorruptCommits(true);
    arms.Run("a corrupted committer answer", Alternating(Kind::Constants, 8), DrawsPerSubmission, [&](const Totals& now, const Totals& base) {
        Require(DrawRecords::Differences(DrawRecords::Field::CommitBindings) != 0 && now.Since(base, Counter::CommitDemoted) != 0, "inject commit: the committer verify did not catch a corrupted buffer binding in the committer answer" + Describe(now, base));
        Require(now.Since(base, Counter::CommitBuilt) != 0, "inject commit: today's path did not rebuild the committer half after a demotion" + Describe(now, base));
    });
    DrawRecords::CorruptCommits(false);
}

namespace DirectReport = AgcDriver::DriverDetail::DirectDraws;
using DirectCounter = DirectReport::Counter;

struct DirectTotals {
    std::array<std::uint64_t, static_cast<std::size_t>(DirectCounter::Count)> counters{};
    std::uint64_t differences = 0;
    std::uint64_t buildDifferences = 0;
    std::uint64_t commitChecks = 0;

    static DirectTotals Now() {
        DirectTotals totals;
        for (std::size_t index = 0; index < totals.counters.size(); ++index) totals.counters[index] = DirectReport::Total(static_cast<DirectCounter>(index));
        for (std::size_t index = 0; index < static_cast<std::size_t>(DrawRecords::Field::Count); ++index) {
            totals.differences += DirectReport::Differences(static_cast<DrawRecords::Field>(index));
            totals.buildDifferences += DirectReport::BuildDifferences(static_cast<DrawRecords::Field>(index));
        }
        totals.commitChecks = DrawRecords::Checked(DrawRecords::Item::Commit);
        return totals;
    }

    std::uint64_t Since(const DirectTotals& base, DirectCounter counter) const { return counters[static_cast<std::size_t>(counter)] - base.counters[static_cast<std::size_t>(counter)]; }
    std::uint64_t Since(const DirectTotals& base, DirectCounter first, DirectCounter last) const {
        std::uint64_t total = 0;
        for (auto index = static_cast<std::size_t>(first); index <= static_cast<std::size_t>(last); ++index) total += Since(base, static_cast<DirectCounter>(index));
        return total;
    }
};

std::string DescribeDirect(const DirectTotals& now, const DirectTotals& base) {
    std::string text;
    for (std::size_t index = 0; index < now.counters.size(); ++index) {
        const auto moved = now.Since(base, static_cast<DirectCounter>(index));
        if (moved != 0) text += std::string("; ") + DirectReport::CounterName(static_cast<DirectCounter>(index)) + " " + std::to_string(moved);
    }
    if (now.commitChecks != base.commitChecks) text += "; committer checks " + std::to_string(now.commitChecks - base.commitChecks);
    return text;
}

// With APS5_DIRECT_BUFFERS set, every set also renders with the switch off first, so the two images
// of a submission are compared through today's; the commit binds the buffers the finisher prepared.
void Direct(bool ring) {
    const bool exact = std::getenv("APS5_DIRECT_DRAWS_EXACT_KEY") != nullptr;
    const bool buffers = RuntimeSwitches::Set(RuntimeSwitch::DirectBuffers);
    const bool inputs = RuntimeSwitches::Set(RuntimeSwitch::DirectInputs);
    const bool repeat = RuntimeSwitches::Set(RuntimeSwitch::DirectRepeat);
    std::vector<State> cycled;
    for (std::uint32_t format = 0; format < RawFormats.size(); ++format) cycled.push_back({Kind::Format, format, 16, 3});
    const std::vector<std::pair<std::string, std::vector<State>>> sets{
        {"constants behind a moving user data pointer", Alternating(Kind::Constants, 4)},
        {"the same constants in draws of other vertex counts", MovingIdentities(Kind::Constants, 4)},
        {"a vertex buffer descriptor moving inside a memory table", Alternating(Kind::Table, 4)},
        {"a fetched vertex buffer moving", Alternating(Kind::Fetch, 4)},
        {"an inline constant buffer moving to a new ring slot", Moving(Kind::Ring, 4)},
        {"a constant buffer and the table in it moving", Moving(Kind::RingInner, 4)},
        {"a buffer format word cycling", cycled},
    };
    const auto base = DirectTotals::Now();
    for (const auto& [name, states] : sets) {
        RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, true);
        RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, false);
        RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, false);
        RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, false);
        std::vector<std::vector<std::uint8_t>> today;
        for (const auto& state : states) today.push_back(Render(state, DrawsPerSubmission));
        RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, true);
        const auto before = DirectTotals::Now();
        if (buffers) {
            RuntimeSwitches::Assign(RuntimeSwitch::DirectBuffers, false);
            for (std::size_t index = 0; index < states.size(); ++index) Require(Render(states[index], DrawsPerSubmission) == today[index], "direct draws without prepared buffers, " + name + ": submission " + std::to_string(index) + " rendered differently from today's path" + DescribeDirect(DirectTotals::Now(), before));
            RuntimeSwitches::Assign(RuntimeSwitch::DirectBuffers, true);
        }
        if (inputs) {
            RuntimeSwitches::Assign(RuntimeSwitch::DirectInputs, false);
            for (std::size_t index = 0; index < states.size(); ++index) Require(Render(states[index], DrawsPerSubmission) == today[index], "direct draws without prepared inputs, " + name + ": submission " + std::to_string(index) + " rendered differently from today's path" + DescribeDirect(DirectTotals::Now(), before));
            RuntimeSwitches::Assign(RuntimeSwitch::DirectInputs, true);
        }
        if (repeat) {
            RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeat, false);
            for (std::size_t index = 0; index < states.size(); ++index) Require(Render(states[index], DrawsPerSubmission) == today[index], "direct draws without repeat commits, " + name + ": submission " + std::to_string(index) + " rendered differently from today's path" + DescribeDirect(DirectTotals::Now(), before));
            RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeat, true);
        }
        for (std::size_t index = 0; index < states.size(); ++index) Require(Render(states[index], DrawsPerSubmission) == today[index], "direct draws, " + name + ": submission " + std::to_string(index) + " rendered differently from today's path" + DescribeDirect(DirectTotals::Now(), before));
        RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, false);
        const auto after = DirectTotals::Now();
        std::cout << "direct draws, " << name << ":" << DescribeDirect(after, before) << '\n';
        if (!ring || name != sets[1].first) continue;
        if (exact) Require(after.Since(before, DirectCounter::ClassesBuilt) != 0, "direct draws, " + name + ": with exact class keys, draws of another vertex count built no class of their own" + DescribeDirect(after, before));
        else Require(after.Since(before, DirectCounter::ClassesBuilt) == 0 && after.Since(before, DirectCounter::FallbackClassNew) == 0, "direct draws, " + name + ": draws of another vertex count did not share the class of the first" + DescribeDirect(after, before));
    }
    const auto now = DirectTotals::Now();
    const auto draws = now.Since(base, DirectCounter::Draws);
    const auto direct = now.Since(base, DirectCounter::Direct);
    const auto fallbacks = now.Since(base, DirectCounter::FallbackShape, DirectCounter::FallbackVerify);
    std::cout << "direct draws: " << draws << " draws, " << direct << " direct, " << fallbacks << " fallbacks, " << now.Since(base, DirectCounter::ClassesBuilt) << " classes built, " << now.Since(base, DirectCounter::StagesBuilt) << " stages built\n";
    if (!ring) return;
    Require(draws != 0 && direct + fallbacks == draws, "direct draws: the direct and fallback draws do not add up to the draws the finisher took" + DescribeDirect(now, base));
    Require(now.differences == base.differences && now.Since(base, DirectCounter::VerifyDiffering) == 0, "direct draws: the verify found differences" + DescribeDirect(now, base));
    Require(now.buildDifferences == base.buildDifferences && now.Since(base, DirectCounter::RefusedDiffers) == 0, "direct draws: a class build found differences" + DescribeDirect(now, base));
    Require(now.Since(base, DirectCounter::ClassesBuilt) != 0, "direct draws: no class was built" + DescribeDirect(now, base));
    const auto verified = now.Since(base, DirectCounter::VerifyChecked);
    if (DirectReport::VerifyEvery() == 1) Require(direct == 0 && verified != 0, "direct draws: with every direct job due for a verify, a class hit committed directly or none was verified" + DescribeDirect(now, base));
    else Require(direct != 0, "direct draws: no draw committed directly" + DescribeDirect(now, base));
    if (DirectReport::VerifyEvery() > 1) Require(verified != 0, "direct draws: the verify checked no class hit" + DescribeDirect(now, base));
    if (DirectReport::VerifyEvery() != 1 && DrawRecords::VerifyEvery() > 1) Require(now.commitChecks > base.commitChecks, "direct draws: the committer verify checked no direct hit" + DescribeDirect(now, base));
    if (inputs && DirectReport::VerifyEvery() != 1 && DrawRecords::VerifyEvery() != 1) {
        const auto prepared = now.Since(base, DirectCounter::InputsPrepared);
        const auto used = now.Since(base, DirectCounter::InputsUsed);
        const auto checked = now.Since(base, DirectCounter::InputsVerifyChecked);
        std::cout << "direct draws: " << prepared << " hits prepared their inputs, " << used << " took them, " << checked << " were verified\n";
        Require(now.Since(base, DirectCounter::InputsVerifyDiffering) == 0, "direct draws: the prepared inputs verify found differences" + DescribeDirect(now, base));
        Require(RuntimeSwitches::Set(RuntimeSwitch::RunFollow) || prepared == used + now.Since(base, DirectCounter::InputsImportChanged, DirectCounter::InputsDropped) + checked, "direct draws: the prepared input hits do not add up to the hits that took them, did not take them and were verified" + DescribeDirect(now, base));
        if (DirectReport::InputVerifyEvery() == 1) Require(used == 0 && checked != 0, "direct draws: with every prepared input hit due for a verify, a commit took prepared inputs or none was verified" + DescribeDirect(now, base));
        else Require(used != 0, "direct draws: no commit took prepared inputs" + DescribeDirect(now, base));
        if (DirectReport::InputVerifyEvery() > 1) Require(checked != 0, "direct draws: the prepared inputs verify checked no hit" + DescribeDirect(now, base));
    }
    if (!buffers || DirectReport::VerifyEvery() == 1 || DrawRecords::VerifyEvery() == 1) return;
    const auto used = now.Since(base, DirectCounter::BuffersUsed);
    const auto checked = now.Since(base, DirectCounter::BuffersVerifyChecked);
    std::cout << "direct draws: " << now.Since(base, DirectCounter::BuffersPrepared) << " hits prepared their buffers, " << used << " bound them, " << checked << " were verified\n";
    Require(now.Since(base, DirectCounter::BuffersVerifyDiffering) == 0 && now.Since(base, DirectCounter::BuffersUnwatched) == 0, "direct draws: the prepared buffers verify found differences" + DescribeDirect(now, base));
    Require(RuntimeSwitches::Set(RuntimeSwitch::RunFollow) || now.Since(base, DirectCounter::BuffersPrepared) == used + now.Since(base, DirectCounter::BuffersPlanDiffers, DirectCounter::BuffersDropped) + checked, "direct draws: the prepared hits do not add up to the hits that bound, did not bind and were verified" + DescribeDirect(now, base));
    if (DirectReport::BufferVerifyEvery() == 1) Require(used == 0 && checked != 0, "direct draws: with every prepared hit due for a verify, a commit bound prepared buffers or none was verified" + DescribeDirect(now, base));
    else Require(used != 0, "direct draws: no commit bound prepared buffers" + DescribeDirect(now, base));
    if (DirectReport::BufferVerifyEvery() > 1) Require(checked != 0, "direct draws: the prepared buffers verify checked no hit" + DescribeDirect(now, base));
}

constexpr std::size_t RepeatConstantsOffset = 0x2000;
constexpr std::size_t RepeatStoreOffset = 0x8000;
constexpr std::size_t RepeatVertexOffset = 0x3000;
constexpr std::uint32_t RepeatDraws = 12;
constexpr std::uint32_t SharedDraws = 13;
constexpr std::uint32_t RepeatRounds = 4;
constexpr std::uint32_t RepeatTimingRounds = 24;
constexpr std::uint32_t AdditiveBlend = 0x40000101u;
constexpr std::array<std::uint32_t, 4> RepeatVertexCounts{3u, 6u, 9u, 6u};
constexpr std::size_t StillSlot = 15;
constexpr std::uint32_t StillDraws = 48;

enum class RepeatScene { Constants, Sets, Ring, Shared, Lend, Still, Mixed };

const char* RepeatSceneName(RepeatScene scene) {
    switch (scene) {
    case RepeatScene::Constants: return "per-draw constants";
    case RepeatScene::Sets: return "per-draw data in both sets";
    case RepeatScene::Ring: return "per-draw constant buffers";
    case RepeatScene::Lend: return "per-draw data beside a constant buffer in place";
    case RepeatScene::Still: return "a constant buffer in place, rewritten per pass";
    case RepeatScene::Mixed: return "per-draw data in both sets, the pushed set at rest every other pass";
    default: return "two classes and a pass break";
    }
}

std::vector<std::uint8_t> RenderScene(RepeatScene scene, std::uint32_t round) {
    auto& blocks = *Scene;
    std::memset(blocks.target.Bytes(0), 0, ColorBytes);
    std::memset(blocks.other.Bytes(0), 0, ColorBytes);
    for (std::uint32_t draw = 0; draw < SharedDraws; ++draw) {
        const float value = 0.0009765625f * static_cast<float>(1 + draw + round * SharedDraws);
        std::memcpy(blocks.data.Bytes(RepeatConstantsOffset + draw * 0x40u), &value, sizeof(value));
    }
    const auto first = blocks.data.At(RepeatConstantsOffset);
    const auto target = blocks.target.At(0);
    const auto other = blocks.other.At(0);
    RegisterWrites registers{{0x8b, 0}, {0x8c, 0}, {0x00b, 3u << 1u}, {0x00c, static_cast<std::uint32_t>(first)}, {0x00d, static_cast<std::uint32_t>(first >> 32u)}, {0x00e, 0u}};
    auto vertex = reinterpret_cast<std::uintptr_t>(PlainVertex.data());
    auto pixel = reinterpret_cast<std::uintptr_t>(ConstantPixel.data());
    if (scene == RepeatScene::Sets || scene == RepeatScene::Lend || scene == RepeatScene::Mixed) {
        for (std::uint32_t draw = 0; draw < RepeatDraws; ++draw) {
            const float shift = 0.0009765625f * static_cast<float>(1 + draw + round * RepeatDraws);
            std::memcpy(blocks.data.Bytes(RepeatVertexOffset + draw * 0x40u), &shift, sizeof(shift));
        }
        const float rest = 0.0009765625f;
        std::memcpy(blocks.data.Bytes(RepeatVertexOffset + RepeatDraws * 0x40u), &rest, sizeof(rest));
        const auto shifts = blocks.data.At(RepeatVertexOffset);
        registers = {{0x8b, 2u << 1u}, {0x8c, static_cast<std::uint32_t>(shifts)}, {0x8d, static_cast<std::uint32_t>(shifts >> 32u)}, {0x00b, 3u << 1u}, {0x00c, static_cast<std::uint32_t>(first)}, {0x00d, static_cast<std::uint32_t>(first >> 32u)}, {0x00e, 0u}};
        vertex = reinterpret_cast<std::uintptr_t>(DataVertex.data());
    }
    if (scene == RepeatScene::Still) {
        const auto slot = RingSlotsOffset + StillSlot * RingSlotBytes;
        for (std::uint32_t word = 0; word < RingSlotBytes / 4u; ++word) {
            const auto value = word >= 4u && word < 8u ? 0u : 0x3a800000u + round * 0x1000u + word;
            std::memcpy(blocks.ring.Bytes(slot + word * 4u), &value, sizeof(value));
        }
        const auto constants = RawDescriptor(blocks.ring.At(slot), ConstantRecords);
        registers = {{0x8b, 0}, {0x8c, 0}, {0x00b, 4u << 1u}, {0x00c, constants[0]}, {0x00d, constants[1]}, {0x00e, constants[2]}, {0x00f, constants[3]}};
        pixel = reinterpret_cast<std::uintptr_t>(RingPixel.data());
    }
    if (scene == RepeatScene::Lend) {
        const auto constants = RawDescriptor(blocks.ring.At(RingSlotsOffset), ConstantRecords);
        const auto shifts = blocks.data.At(RepeatVertexOffset);
        registers = {{0x8b, 2u << 1u}, {0x8c, static_cast<std::uint32_t>(shifts)}, {0x8d, static_cast<std::uint32_t>(shifts >> 32u)}, {0x00b, 4u << 1u}, {0x00c, constants[0]}, {0x00d, constants[1]}, {0x00e, constants[2]}, {0x00f, constants[3]}};
        pixel = reinterpret_cast<std::uintptr_t>(RingPixel.data());
    }
    if (scene == RepeatScene::Ring) {
        const auto constants = RawDescriptor(blocks.ring.At(RingSlotsOffset + (round % 6) * RingSlotBytes), ConstantRecords);
        registers = {{0x8b, 0}, {0x8c, 0}, {0x00b, 4u << 1u}, {0x00c, constants[0]}, {0x00d, constants[1]}, {0x00e, constants[2]}, {0x00f, constants[3]}};
        pixel = reinterpret_cast<std::uintptr_t>(RingPixel.data());
    }
    const auto draws = scene == RepeatScene::Shared ? SharedDraws : scene == RepeatScene::Still ? StillDraws : RepeatDraws;
    SubmitDraws(vertex, pixel, {{0x242, 4}}, registers, blocks.target, draws, 3, AdditiveBlend, [&](std::vector<std::uint32_t>& commands, std::uint32_t draw) -> std::uint32_t {
        if (scene == RepeatScene::Ring || scene == RepeatScene::Still) return RepeatVertexCounts[(draw + round) % RepeatVertexCounts.size()];
        if (scene == RepeatScene::Lend) {
            const auto shift = blocks.data.At(RepeatVertexOffset + draw * 0x40u);
            Append(commands, 0x76, {0x8c, static_cast<std::uint32_t>(shift)});
            Append(commands, 0x76, {0x8d, static_cast<std::uint32_t>(shift >> 32u)});
            return 3;
        }
        const auto pointer = blocks.data.At(RepeatConstantsOffset + draw * 0x40u);
        Append(commands, 0x76, {0x00c, static_cast<std::uint32_t>(pointer)});
        Append(commands, 0x76, {0x00d, static_cast<std::uint32_t>(pointer >> 32u)});
        if (scene == RepeatScene::Sets || scene == RepeatScene::Mixed) {
            const auto shift = blocks.data.At(RepeatVertexOffset + (scene == RepeatScene::Mixed && round % 2u == 1u ? RepeatDraws : draw) * 0x40u);
            Append(commands, 0x76, {0x8c, static_cast<std::uint32_t>(shift)});
            Append(commands, 0x76, {0x8d, static_cast<std::uint32_t>(shift >> 32u)});
            return 3;
        }
        if (scene == RepeatScene::Shared) {
            const bool second = (draw >= 3 && draw < 6) || draw >= 10;
            const bool away = draw == 6;
            Append(commands, 0x69, {0x1e0, second ? 0u : AdditiveBlend});
            Append(commands, 0x69, {0x318, static_cast<std::uint32_t>((away ? other : target) >> 8u)});
            Append(commands, 0x69, {0x390, static_cast<std::uint32_t>((away ? other : target) >> 40u)});
            return 3;
        }
        return RepeatVertexCounts[(draw + round) % RepeatVertexCounts.size()];
    });
    auto image = Image(blocks.target);
    if (scene == RepeatScene::Shared) {
        const auto away = Image(blocks.other);
        image.insert(image.end(), away.begin(), away.end());
    }
    return image;
}

std::vector<std::uint8_t> RenderRepeats(std::uint32_t round) {
    return RenderScene(RepeatScene::Constants, round);
}

namespace RepeatHooks {

std::atomic<int> fired{0};

void QueueStore(AgcDriver::DrawRecordCommit&) {
    if (auto* recorder = AgcDriver::Graphics::Recorder::Active(); recorder != nullptr) {
        const auto address = Scene->ring.At(RepeatStoreOffset);
        recorder->QueueKeyStore(VK_NULL_HANDLE, 0, 0, nullptr, address, address + 4u);
    }
    fired.fetch_add(1);
}

void QueueTouchingStore(AgcDriver::DrawRecordCommit& record) {
    auto* recorder = AgcDriver::Graphics::Recorder::Active();
    if (recorder != nullptr && record.repeat != nullptr && record.repeat->resources != nullptr) {
        const auto reads = record.repeat->resources->InPlaceReads();
        if (!reads.empty()) {
            recorder->QueueKeyStore(VK_NULL_HANDLE, 0, 0, nullptr, reads.front().first, reads.front().first + 4u);
            fired.fetch_add(1);
        }
    }
}

void Collect(AgcDriver::DrawRecordCommit&) {
    AgcDriver::GuestMemory::BumpCollectEpoch();
    fired.fetch_add(1);
}

void Refresh(AgcDriver::DrawRecordCommit& record) {
    if (!record.pass.versions.empty()) ++record.pass.versions.front();
    fired.fetch_add(1);
}

void Batch(AgcDriver::DrawRecordCommit&) {
    if (auto* recorder = AgcDriver::Graphics::Recorder::Active(); recorder != nullptr) recorder->Submit();
    fired.fetch_add(1);
}

void Toggle(AgcDriver::DrawRecordCommit&) {
    const bool set = RuntimeSwitches::Set(RuntimeSwitch::DirectRepeat);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeat, !set);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeat, set);
    fired.fetch_add(1);
}

}

struct RepeatSwitches {
    bool sets = false;
    bool stores = false;
    bool shared = false;

    void Apply() const {
        RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeatSets, sets);
        RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeatStores, stores);
        RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeatShared, shared);
    }
};

void Repeats(bool ring) {
    using AgcDriver::Graphics::SetRepeatHookForTest;
    const auto verifyEvery = DirectReport::RepeatVerifyEvery();
    const bool repeatEnvironment = RuntimeSwitches::Set(RuntimeSwitch::DirectRepeat);
    const RepeatSwitches environment{RuntimeSwitches::Set(RuntimeSwitch::DirectRepeatSets), RuntimeSwitches::Set(RuntimeSwitch::DirectRepeatStores), RuntimeSwitches::Set(RuntimeSwitch::DirectRepeatShared)};
    const bool passTargets = RuntimeSwitches::Set(RuntimeSwitch::DirectPassTargets);
    const bool committed = ring && DirectReport::VerifyEvery() != 1 && DrawRecords::VerifyEvery() != 1 && RuntimeSwitches::Set(RuntimeSwitch::DirectBuffers) && RuntimeSwitches::Set(RuntimeSwitch::DirectInputs) && DirectReport::BufferVerifyEvery() == 0 && DirectReport::InputVerifyEvery() == 0;
    RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, true);
    RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, false);
    RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, false);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectPassTargets, true);
    const auto perCommit = [](const DirectTotals& now, const DirectTotals& base, DirectCounter nanoseconds, DirectCounter timed) {
        const auto count = now.Since(base, timed);
        return count == 0 ? 0.0 : static_cast<double>(now.Since(base, nanoseconds)) / static_cast<double>(count) / 1000.0;
    };
    for (const auto scene : {RepeatScene::Constants, RepeatScene::Sets, RepeatScene::Ring, RepeatScene::Shared}) {
        const std::string name = RepeatSceneName(scene);
        RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, false);
        RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeat, false);
        std::vector<std::vector<std::uint8_t>> today;
        for (std::uint32_t round = 0; round < RepeatRounds; ++round) today.push_back(RenderScene(scene, round));
        RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, true);
        const auto render = [&](const RepeatSwitches& switches, const std::string& what) {
            switches.Apply();
            const auto before = DirectTotals::Now();
            for (std::uint32_t round = 0; round < RepeatRounds; ++round) Require(RenderScene(scene, round) == today[round], "repeat commits, " + name + ", " + what + ": round " + std::to_string(round) + " rendered differently from today's path" + DescribeDirect(DirectTotals::Now(), before));
            const auto after = DirectTotals::Now();
            std::cout << "repeat commits, " << name << ", " << what << ": " << after.Since(before, DirectCounter::RepeatTaken) << " taken, " << after.Since(before, DirectCounter::RepeatVerifyChecked) << " verified" << DescribeDirect(after, before) << '\n';
            Require(after.Since(before, DirectCounter::RepeatVerifyDiffering) == 0 && after.Since(before, DirectCounter::RepeatRefuted) == 0, "repeat commits, " + name + ", " + what + ": the verify found differences or refuted claims" + DescribeDirect(after, before));
            return std::pair{before, after};
        };
        RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeat, false);
        static_cast<void>(render({}, "without repeats"));
        RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeat, true);
        const auto [plainBefore, plainAfter] = render({}, "without the sub-switches");
        const auto [allBefore, allAfter] = render({true, true, true}, "with every sub-switch");
        if (!committed) continue;
        const auto taken = allAfter.Since(allBefore, DirectCounter::RepeatTaken);
        const auto checked = allAfter.Since(allBefore, DirectCounter::RepeatVerifyChecked);
        Require(allAfter.Since(allBefore, DirectCounter::RepeatBlocks) != 0, "repeat commits, " + name + ": no full commit built a repeat block" + DescribeDirect(allAfter, allBefore));
        if (verifyEvery == 1) Require(taken == 0 && checked != 0, "repeat commits, " + name + ": with every repeat due for a verify, a repeat was taken or none was verified" + DescribeDirect(allAfter, allBefore));
        else Require(taken != 0, "repeat commits, " + name + ": no repeat was taken" + DescribeDirect(allAfter, allBefore));
        if (verifyEvery > 1) Require(checked != 0, "repeat commits, " + name + ": the verify checked no repeat" + DescribeDirect(allAfter, allBefore));
        if (scene == RepeatScene::Sets) {
            Require(plainAfter.Since(plainBefore, DirectCounter::RepeatClassSets) != 0, "repeat commits, " + name + ": without APS5_DIRECT_REPEAT_SETS no class was refused for its second set" + DescribeDirect(plainAfter, plainBefore));
            if (verifyEvery != 1) Require(allAfter.Since(allBefore, DirectCounter::RepeatTakenSets) != 0, "repeat commits, " + name + ": with APS5_DIRECT_REPEAT_SETS no repeat wrote the second set" + DescribeDirect(allAfter, allBefore));
        }
        if (scene == RepeatScene::Shared && verifyEvery != 1) Require(allAfter.Since(allBefore, DirectCounter::RepeatTakenShared) != 0 && plainAfter.Since(plainBefore, DirectCounter::RepeatTakenShared) == 0, "repeat commits, " + name + ": no repeat began under the shared pass memo with APS5_DIRECT_REPEAT_SHARED, or one did without it" + DescribeDirect(allAfter, allBefore));
        const auto expectHook = [&](const std::string& what, AgcDriver::Graphics::RepeatHookForTest hook, DirectCounter counter, const RepeatSwitches& switches) {
            switches.Apply();
            const auto before = DirectTotals::Now();
            RepeatHooks::fired.store(0);
            SetRepeatHookForTest(hook);
            const auto image = RenderScene(scene, 0);
            SetRepeatHookForTest(nullptr);
            const auto after = DirectTotals::Now();
            Require(RepeatHooks::fired.load() == 1, "repeat commits, " + name + ", " + what + ": the hook did not run once before a repeat");
            Require(after.Since(before, counter) != 0, "repeat commits, " + name + ", " + what + ": no repeat was counted as " + DirectReport::CounterName(counter) + DescribeDirect(after, before));
            Require(image == today[0], "repeat commits, " + name + ", " + what + ": the pass rendered differently from today's path" + DescribeDirect(after, before));
            std::cout << "repeat commits, " << name << ", " << what << ": counted as " << DirectReport::CounterName(counter) << '\n';
        };
        if (scene == RepeatScene::Constants) {
            expectHook("a store queued between two draws", &RepeatHooks::QueueStore, DirectCounter::RepeatStores, {});
            expectHook("a collect between two draws", &RepeatHooks::Collect, DirectCounter::RepeatEpoch, {});
            expectHook("a target refreshed between two draws", &RepeatHooks::Refresh, DirectCounter::RepeatTarget, {});
            expectHook("a new batch between two draws", &RepeatHooks::Batch, DirectCounter::RepeatBatch, {});
            expectHook("a switch toggled between two draws", &RepeatHooks::Toggle, DirectCounter::RepeatShape, {});
        }
        if (scene == RepeatScene::Ring) {
            if (verifyEvery != 1) expectHook("a store queued away from the class, with APS5_DIRECT_REPEAT_STORES", &RepeatHooks::QueueStore, DirectCounter::RepeatTakenStores, {true, true, true});
            expectHook("a store queued over the class's reads, with APS5_DIRECT_REPEAT_STORES", &RepeatHooks::QueueTouchingStore, DirectCounter::RepeatStores, {true, true, true});
        }
        if (std::getenv("APS5_PIPELINE_STATS") != nullptr && verifyEvery != 1) {
            RepeatSwitches{}.Apply();
            RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeat, false);
            const auto fullBase = DirectTotals::Now();
            for (std::uint32_t round = 0; round < RepeatTimingRounds; ++round) static_cast<void>(RenderScene(scene, round % RepeatRounds));
            const auto full = DirectTotals::Now();
            RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeat, true);
            RepeatSwitches{true, true, true}.Apply();
            for (std::uint32_t round = 0; round < RepeatTimingRounds; ++round) static_cast<void>(RenderScene(scene, round % RepeatRounds));
            const auto repeated = DirectTotals::Now();
            std::cout << "repeat commit timing, " << name << ": full route " << perCommit(full, fullBase, DirectCounter::RepeatFullNanoseconds, DirectCounter::RepeatFullTimed) << " us per commit held by the record (" << full.Since(fullBase, DirectCounter::RepeatFullTimed) << " sampled), repeat " << perCommit(repeated, full, DirectCounter::RepeatNanoseconds, DirectCounter::RepeatTimed) << " us (" << repeated.Since(full, DirectCounter::RepeatTimed) << " sampled; with a written set " << perCommit(repeated, full, DirectCounter::RepeatSetsNanoseconds, DirectCounter::RepeatSetsTimed) << " us, " << repeated.Since(full, DirectCounter::RepeatSetsTimed) << " sampled; under the shared hold " << perCommit(repeated, full, DirectCounter::RepeatSharedNanoseconds, DirectCounter::RepeatSharedTimed) << " us, " << repeated.Since(full, DirectCounter::RepeatSharedTimed) << " sampled)\n";
            Require(full.Since(fullBase, DirectCounter::RepeatFullTimed) != 0 && repeated.Since(full, DirectCounter::RepeatTimed) != 0, "repeat commit timing, " + name + ": no full commit or no repeat was timed" + DescribeDirect(repeated, fullBase));
        }
    }
    environment.Apply();
    RuntimeSwitches::Assign(RuntimeSwitch::DirectPassTargets, passTargets);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, false);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeat, repeatEnvironment);
}

namespace BindingHooks {

std::atomic<int> fired{0};

std::vector<std::pair<std::uint64_t, std::uint64_t>> InPlaceReads(AgcDriver::DrawRecordCommit& record) {
    if (record.recipe == nullptr) return {};
    const auto resources = record.recipe->templateRef.lock();
    if (resources == nullptr) return {};
    return resources->InPlaceReads();
}

void PendingWrite(AgcDriver::DrawRecordCommit& record) {
    auto* recorder = AgcDriver::Graphics::Recorder::Active();
    const auto reads = InPlaceReads(record);
    if (recorder == nullptr || reads.empty()) return;
    for (const auto& [begin, end] : reads) recorder->NotePendingWrite(begin, static_cast<std::size_t>(end - begin));
    fired.fetch_add(1);
}

void QueuedStore(AgcDriver::DrawRecordCommit& record) {
    auto* recorder = AgcDriver::Graphics::Recorder::Active();
    const auto reads = InPlaceReads(record);
    if (recorder == nullptr || reads.empty()) return;
    for (const auto& [begin, end] : reads) recorder->QueueKeyStore(VK_NULL_HANDLE, 0, 0, nullptr, begin, end);
    fired.fetch_add(1);
}

}

namespace ValidationHooks {

std::atomic<int> calls{0};
std::atomic<int> fired{0};
std::atomic<int> kind{0};

void Event(AgcDriver::DrawRecordCommit& record) {
    if (fired.load() != 0 || calls.fetch_add(1) < 1) return;
    auto* recorder = AgcDriver::Graphics::Recorder::Active();
    const auto reads = BindingHooks::InPlaceReads(record);
    if (recorder == nullptr || reads.empty()) return;
    const auto which = kind.load();
    for (const auto& [begin, end] : reads) {
        const auto bytes = static_cast<std::size_t>(end - begin);
        if (which == 0) {
            recorder->NotePendingWrite(begin, bytes);
        } else if (which == 1) {
            recorder->QueueKeyStore(VK_NULL_HANDLE, 0, 0, nullptr, begin, end);
        } else {
            std::vector<std::byte> current(bytes);
            std::memcpy(current.data(), reinterpret_cast<const void*>(begin), bytes);
            AgcDriver::GuestMemory::Write(begin, current);
        }
    }
    if (which == 2) AgcDriver::GuestMemory::BumpCollectEpoch();
    fired.fetch_add(1);
}

}

std::uint64_t ValidationSum(const AgcDriver::Graphics::WriteValidation::Totals& totals, AgcDriver::Graphics::WriteValidation::Outcome outcome) {
    std::uint64_t sum = 0;
    for (const auto& row : totals.outcomes) sum += row[static_cast<std::size_t>(outcome)];
    return sum;
}

std::uint64_t ValidationDiffering(const AgcDriver::Graphics::WriteValidation::Totals& totals) {
    std::uint64_t sum = 0;
    for (const auto count : totals.differing) sum += count;
    return sum;
}

std::string DescribeValidation(const AgcDriver::Graphics::WriteValidation::Totals& after, const AgcDriver::Graphics::WriteValidation::Totals& before) {
    namespace Validation = AgcDriver::Graphics::WriteValidation;
    std::string text;
    for (std::size_t test = 0; test < Validation::TestCount; ++test) {
        std::string row;
        for (std::size_t outcome = 0; outcome < Validation::OutcomeCount; ++outcome) {
            const auto count = after.outcomes[test][outcome] - before.outcomes[test][outcome];
            if (count == 0) continue;
            row += (row.empty() ? "" : ", ") + std::string(Validation::OutcomeName(static_cast<Validation::Outcome>(outcome))) + " " + std::to_string(count);
        }
        const auto verified = after.verified[test] - before.verified[test];
        if (verified != 0) row += (row.empty() ? "" : ", ") + std::string("verified ") + std::to_string(verified);
        if (!row.empty()) text += std::string("; ") + Validation::TestName(static_cast<Validation::Test>(test)) + ": " + row;
    }
    return text;
}

void WriteValidationEvents(bool ring) {
    namespace Validation = AgcDriver::Graphics::WriteValidation;
    using AgcDriver::Graphics::SetExpressHookForTest;
    const bool validationEnvironment = RuntimeSwitches::Set(RuntimeSwitch::WriteValidation);
    const bool repeatEnvironment = RuntimeSwitches::Set(RuntimeSwitch::DirectRepeat);
    const bool passTargets = RuntimeSwitches::Set(RuntimeSwitch::DirectPassTargets);
    const bool committed = ring && RuntimeSwitches::Set(RuntimeSwitch::ExpressCommits) && RuntimeSwitches::Set(RuntimeSwitch::CommitBindings);
    RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, true);
    RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, false);
    RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, false);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeat, false);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectPassTargets, true);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, true);
    const auto scene = RepeatScene::Ring;
    const std::string name = RepeatSceneName(scene);
    RuntimeSwitches::Assign(RuntimeSwitch::WriteValidation, false);
    Require(Validation::Session() == 0, "write validation, " + name + ": the session stayed open with the switch off");
    std::vector<std::vector<std::uint8_t>> today;
    for (std::uint32_t round = 0; round < RepeatRounds; ++round) today.push_back(RenderScene(scene, round));
    RuntimeSwitches::Assign(RuntimeSwitch::WriteValidation, true);
    const bool active = Validation::Session() != 0;
    const auto before = Validation::Collect();
    for (std::uint32_t pass = 0; pass < 2; ++pass) {
        for (std::uint32_t round = 0; round < RepeatRounds; ++round) Require(RenderScene(scene, round) == today[round], "write validation, " + name + ": round " + std::to_string(round) + " rendered differently from today's path with the switch on");
    }
    const auto after = Validation::Collect();
    std::cout << "write validation, " << name << ": session " << Validation::Session() << DescribeValidation(after, before) << '\n';
    Require(ValidationDiffering(after) == ValidationDiffering(before), "write validation, " + name + ": the verify found a test answered by generations that today's test refused" + DescribeValidation(after, before));
    if (active && committed) Require(ValidationSum(after, Validation::Outcome::Answered) + ValidationSum(after, Validation::Outcome::Armed) != ValidationSum(before, Validation::Outcome::Answered) + ValidationSum(before, Validation::Outcome::Armed), "write validation, " + name + ": no test was answered by generations or proved with the switch on" + DescribeValidation(after, before));
    const auto expectEvent = [&](int which, const std::string& what) {
        ValidationHooks::calls.store(0);
        ValidationHooks::fired.store(0);
        ValidationHooks::kind.store(which);
        const auto base = Validation::Collect();
        SetExpressHookForTest(&ValidationHooks::Event);
        const auto image = RenderScene(scene, 1);
        SetExpressHookForTest(nullptr);
        const auto done = Validation::Collect();
        Require(image == today[1], "write validation, " + name + ", " + what + ": the pass rendered differently from today's path" + DescribeValidation(done, base));
        Require(ValidationDiffering(done) == ValidationDiffering(base), "write validation, " + name + ", " + what + ": a test answered by generations after the event was refused by today's test" + DescribeValidation(done, base));
        Require(RenderScene(scene, 2) == today[2], "write validation, " + name + ", " + what + ": the next pass rendered differently from today's path");
        const auto next = Validation::Collect();
        Require(ValidationDiffering(next) == ValidationDiffering(base), "write validation, " + name + ", " + what + ": a test of the next pass answered by generations was refused by today's test" + DescribeValidation(next, done));
        if (active && committed) Require(ValidationHooks::fired.load() == 1, "write validation, " + name + ", " + what + ": the hook did not run once between two express commits");
        std::cout << "write validation, " << name << ", " << what << ": " << (ValidationHooks::fired.load() == 1 ? "raised between two express commits" : "the hook did not fire") << DescribeValidation(done, base) << '\n';
    };
    expectEvent(0, "a GPU write registered over the template's in-place reads");
    expectEvent(1, "a key store queued over the template's in-place reads");
    expectEvent(2, "a store over the template's in-place reads and a collect");
    RuntimeSwitches::Assign(RuntimeSwitch::WriteValidation, validationEnvironment);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeat, repeatEnvironment);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectPassTargets, passTargets);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, false);
}

std::size_t lingeringChunks() {
    std::size_t threads = RuntimeSwitches::Set(RuntimeSwitch::SiblingFinishers) ? 5u : 1u;
    for (const auto id : {RuntimeSwitch::SecondFinisher, RuntimeSwitch::ThirdFinisher, RuntimeSwitch::FourthFinisher, RuntimeSwitch::FifthFinisher}) threads += RuntimeSwitches::Set(id) ? 1u : 0u;
    return 2u * threads;
}
constexpr std::size_t SmallChunkBytes = 128;
constexpr std::size_t RingChunkBytes = 1536;

std::size_t LooseChunks() {
    const auto placed = AgcDriver::Graphics::ShaderResources::PlacedUploadChunks();
    const auto ring = static_cast<std::size_t>(AgcDriver::Graphics::ShaderResources::UploadRingTotals().chunks);
    return placed > ring ? placed - ring : 0;
}

std::size_t PlacedChunksAfterIdle(std::size_t bound) {
    for (std::uint32_t attempt = 0; attempt < 400 && LooseChunks() > bound; ++attempt) {
        AgcDriverWaitIdle_nid_postfix();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return LooseChunks();
}

void CommitBindings(bool ring) {
    using AgcDriver::Graphics::SetCommitBindingsHookForTest;
    using AgcDriver::Graphics::ShaderResources;
    const auto verifyEvery = AgcDriver::Graphics::CommitBindingsVerifyEvery();
    const bool stampsEnvironment = RuntimeSwitches::Set(RuntimeSwitch::CommitBindingsStamps);
    const bool setsEnvironment = RuntimeSwitches::Set(RuntimeSwitch::CommitBindingsSets);
    const bool dynamicEnvironment = RuntimeSwitches::Set(RuntimeSwitch::CommitBindingsDynamic);
    const bool ownSets = RuntimeSwitches::Set(RuntimeSwitch::OwnDrawSets);
    RuntimeSwitches::Assign(RuntimeSwitch::OwnDrawSets, true);
    const bool repeatEnvironment = RuntimeSwitches::Set(RuntimeSwitch::DirectRepeat);
    const bool passTargets = RuntimeSwitches::Set(RuntimeSwitch::DirectPassTargets);
    const bool committed = ring && DirectReport::VerifyEvery() != 1 && DrawRecords::VerifyEvery() != 1 && RuntimeSwitches::Set(RuntimeSwitch::DirectBuffers) && DirectReport::BufferVerifyEvery() == 0;
    RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, true);
    RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, false);
    RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, false);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeat, false);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectPassTargets, true);
    const auto perDraw = [](const DirectTotals& now, const DirectTotals& base, DirectCounter nanoseconds, DirectCounter timed) {
        const auto count = now.Since(base, timed);
        return count == 0 ? 0.0 : static_cast<double>(now.Since(base, nanoseconds)) / static_cast<double>(count);
    };
    for (const auto scene : {RepeatScene::Constants, RepeatScene::Sets, RepeatScene::Ring, RepeatScene::Lend, RepeatScene::Mixed}) {
        const std::string name = RepeatSceneName(scene);
        RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, false);
        RuntimeSwitches::Assign(RuntimeSwitch::CommitBindings, false);
        std::vector<std::vector<std::uint8_t>> today;
        for (std::uint32_t round = 0; round < RepeatRounds; ++round) today.push_back(RenderScene(scene, round));
        RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, true);
        const auto render = [&](bool bindings, bool stamps, bool sets, bool dynamic, const std::string& what) {
            RuntimeSwitches::Assign(RuntimeSwitch::CommitBindings, bindings);
            RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsStamps, stamps);
            RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsSets, sets);
            RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsDynamic, dynamic);
            const auto before = DirectTotals::Now();
            for (std::uint32_t round = 0; round < RepeatRounds; ++round) Require(RenderScene(scene, round) == today[round], "commit bindings, " + name + ", " + what + ": round " + std::to_string(round) + " rendered differently from today's path" + DescribeDirect(DirectTotals::Now(), before));
            const auto after = DirectTotals::Now();
            std::cout << "commit bindings, " << name << ", " << what << ": " << after.Since(before, DirectCounter::BindingsTaken) << " finisher slices bound, " << after.Since(before, DirectCounter::BindingsStamped) << " unmoved tests answered by stamp, " << after.Since(before, DirectCounter::BindingsSetsLent) << " sets lent, " << after.Since(before, DirectCounter::BindingsVerifyChecked) << " verified, " << ShaderResources::PlacedUploadChunks() << " finisher chunks alive" << DescribeDirect(after, before) << '\n';
            Require(after.Since(before, DirectCounter::BindingsVerifyDiffering) == 0 && after.Since(before, DirectCounter::BindingsRefuted) == 0, "commit bindings, " + name + ", " + what + ": the verify found differences or refuted stamps" + DescribeDirect(after, before));
            Require(PlacedChunksAfterIdle(lingeringChunks()) <= lingeringChunks(), "commit bindings, " + name + ", " + what + ": " + std::to_string(ShaderResources::PlacedUploadChunks()) + " finisher upload chunks are still alive two seconds after the batches completed");
            return std::pair{before, after};
        };
        static_cast<void>(render(false, false, false, false, "without the switch"));
        const auto [placedBefore, placedAfter] = render(true, false, false, false, "with finisher slices");
        const auto [stampedBefore, stampedAfter] = render(true, true, true, false, "with finisher slices, stamps and lent sets");
        const auto [dynamicBefore, dynamicAfter] = render(true, true, true, true, "with every sub-switch");
        if (committed && scene == RepeatScene::Sets) Require(dynamicAfter.Since(dynamicBefore, DirectCounter::BindingsDynamicPushedBound) != 0, "commit bindings, " + name + ": no pushed draw with a written second set bound dynamic sets" + DescribeDirect(dynamicAfter, dynamicBefore));
        if (committed && scene == RepeatScene::Mixed) Require(dynamicAfter.Since(dynamicBefore, DirectCounter::BindingsDynamicBound) != 0 && dynamicAfter.Since(dynamicBefore, DirectCounter::BindingsDynamicPushedBound) != 0 && dynamicAfter.Since(dynamicBefore, DirectCounter::BindingsLeftUnpushedPushHeld) == 0, "commit bindings, " + name + ": a record that holds a push pipeline did not bind dynamic sets both when its pushed set moved and when only its other set did" + DescribeDirect(dynamicAfter, dynamicBefore));
        if (committed && verifyEvery != 0 && dynamicAfter.Since(dynamicBefore, DirectCounter::BindingsDynamicBound) + dynamicAfter.Since(dynamicBefore, DirectCounter::BindingsDynamicPushedBound) != 0) Require(dynamicAfter.Since(dynamicBefore, DirectCounter::BindingsVerifyChecked) != 0, "commit bindings, " + name + ": the verify checked no commit with dynamic sets" + DescribeDirect(dynamicAfter, dynamicBefore));
        const bool fragmentsOn = std::getenv("APS5_COMMIT_FRAGMENTS") != nullptr;
        if (committed && fragmentsOn) Require(placedAfter.Since(placedBefore, DirectCounter::FragmentsSpliced) + placedAfter.Since(placedBefore, DirectCounter::FragmentsReplayed) + placedAfter.Since(placedBefore, DirectCounter::FragmentsVerifyChecked) != 0 && placedAfter.Since(placedBefore, DirectCounter::FragmentsVerifyDiffering) == 0, "commit fragments, " + name + ": no commit used a finisher fragment, or the verify found a difference" + DescribeDirect(placedAfter, placedBefore));
        if (committed && scene == RepeatScene::Lend && !fragmentsOn) Require(stampedAfter.Since(stampedBefore, DirectCounter::BindingsSetsLent) != 0,"commit bindings, " + name + ": no draw borrowed the set an earlier draw wrote" + DescribeDirect(stampedAfter, stampedBefore));
        if (!committed) continue;
        Require(placedAfter.Since(placedBefore, DirectCounter::BindingsTaken) != 0 && (fragmentsOn || placedAfter.Since(placedBefore, DirectCounter::BindingsCopied) == 0),"commit bindings, " + name + ": no commit bound the finisher's slice, or one copied the block" + DescribeDirect(placedAfter, placedBefore));
        Require(fragmentsOn || (placedAfter.Since(placedBefore, DirectCounter::BindingsBytes) != 0 && placedAfter.Since(placedBefore, DirectCounter::BindingsObjects) != 0),"commit bindings, " + name + ": no bytes were left uncopied or no object unallocated" + DescribeDirect(placedAfter, placedBefore));
        if (verifyEvery != 0) Require(placedAfter.Since(placedBefore, DirectCounter::BindingsVerifyChecked) != 0, "commit bindings, " + name + ": the verify checked no commit" + DescribeDirect(placedAfter, placedBefore));
        if (scene == RepeatScene::Sets && !fragmentsOn) Require(placedAfter.Since(placedBefore, DirectCounter::BindingsTakenWritten) != 0,"commit bindings, " + name + ": no commit with a written set bound the finisher's slice" + DescribeDirect(placedAfter, placedBefore));
        if (scene == RepeatScene::Ring && std::getenv("APS5_EXPRESS_COMMITS") == nullptr) Require(stampedAfter.Since(stampedBefore, DirectCounter::BindingsStamped) != 0, "commit bindings, " + name + ": no unmoved test was answered by stamp" + DescribeDirect(stampedAfter, stampedBefore));
        if (scene == RepeatScene::Ring) {
            const auto expectHook = [&](const std::string& what, AgcDriver::Graphics::CommitBindingsHookForTest hook, bool refused) {
                RuntimeSwitches::Assign(RuntimeSwitch::CommitBindings, true);
                RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsStamps, true);
                RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsDynamic, false);
                const auto before = DirectTotals::Now();
                BindingHooks::fired.store(0);
                SetCommitBindingsHookForTest(hook);
                const auto image = RenderScene(scene, 1);
                SetCommitBindingsHookForTest(nullptr);
                const auto after = DirectTotals::Now();
                Require(BindingHooks::fired.load() == 1, "commit bindings, " + name + ", " + what + ": the hook did not run once before a commit");
                if (refused) Require(after.Since(before, DirectCounter::BindingsHazard) != 0 && (fragmentsOn || after.Since(before, DirectCounter::BindingsReturned) != 0), "commit bindings, " + name + ", " + what + ": no commit refused its finisher slice for an unmoved-range hazard and returned it" + DescribeDirect(after, before));
                else Require(after.Since(before, DirectCounter::BindingsHazard) == 0 && (fragmentsOn || after.Since(before, DirectCounter::BindingsTaken) == after.Since(before, DirectCounter::BuffersUsed)),"commit bindings, " + name + ", " + what + ": a commit that binds the prepared buffers today did not bind the finisher's slice" + DescribeDirect(after, before));
                Require(after.Since(before, DirectCounter::BindingsVerifyDiffering) == 0 && after.Since(before, DirectCounter::BindingsRefuted) == 0, "commit bindings, " + name + ", " + what + ": the verify found differences or refuted stamps" + DescribeDirect(after, before));
                Require(image == today[1], "commit bindings, " + name + ", " + what + ": the pass rendered differently from today's path" + DescribeDirect(after, before));
                RuntimeSwitches::Assign(RuntimeSwitch::CommitBindings, false);
                Require(RenderScene(scene, 2) == today[2], "commit bindings, " + name + ", " + what + ": the next pass rendered differently from today's path");
                Require(PlacedChunksAfterIdle(lingeringChunks()) <= lingeringChunks(), "commit bindings, " + name + ", " + what + ": " + std::to_string(ShaderResources::PlacedUploadChunks()) + " finisher upload chunks are still alive two seconds after the next batch completed");
                std::cout << "commit bindings, " << name << ", " << what << ": " << (refused ? "refused and returned" : "bound as today's prepared buffers") << DescribeDirect(after, before) << '\n';
            };
            expectHook("a pending write over a selected in-place buffer before the commit", &BindingHooks::PendingWrite, true);
            expectHook("a key store queued over a selected in-place buffer before the commit", &BindingHooks::QueuedStore, false);
        }
        if (std::getenv("APS5_PIPELINE_STATS") != nullptr && verifyEvery == 0) {
            const auto timing = [&](bool bindings) {
                RuntimeSwitches::Assign(RuntimeSwitch::CommitBindings, bindings);
                RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsStamps, bindings);
                RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsSets, bindings);
                RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsDynamic, bindings);
                const auto before = DirectTotals::Now();
                for (std::uint32_t round = 0; round < RepeatTimingRounds; ++round) static_cast<void>(RenderScene(scene, round % RepeatRounds));
                return std::pair{before, DirectTotals::Now()};
            };
            constexpr std::uint32_t TimingRuns = 7;
            std::array<double, 6> best{};
            best.fill(std::numeric_limits<double>::max());
            const auto keepBest = [&](std::size_t at, double value) {
                if (value != 0.0) best[at] = std::min(best[at], value);
            };
            for (std::uint32_t run = 0; run < TimingRuns; ++run) {
                const auto [todayBefore, todayAfter] = timing(false);
                const auto [newBefore, newAfter] = timing(true);
                keepBest(0, perDraw(todayAfter, todayBefore, DirectCounter::BindingsTodayPushedNanoseconds, DirectCounter::BindingsTodayPushedTimed));
                keepBest(1, perDraw(todayAfter, todayBefore, DirectCounter::BindingsTodayWrittenNanoseconds, DirectCounter::BindingsTodayWrittenTimed));
                keepBest(2, perDraw(newAfter, newBefore, DirectCounter::BindingsNewPushedNanoseconds, DirectCounter::BindingsNewPushedTimed));
                keepBest(3, perDraw(newAfter, newBefore, DirectCounter::BindingsNewWrittenNanoseconds, DirectCounter::BindingsNewWrittenTimed));
                keepBest(4, perDraw(todayAfter, todayBefore, DirectCounter::BindingsKeepNanoseconds, DirectCounter::BindingsKeepTimed));
                keepBest(5, perDraw(newAfter, newBefore, DirectCounter::BindingsKeepNanoseconds, DirectCounter::BindingsKeepTimed));
            }
            const auto shown = [&](std::size_t at) { return best[at] == std::numeric_limits<double>::max() ? 0.0 : best[at]; };
            std::cout << "commit bindings timing, " << name << ": ns per binding step (best of " << TimingRuns << " runs), today's route pushed " << shown(0) << ", with a written set " << shown(1) << "; with the switches pushed " << shown(2) << ", with a written set " << shown(3) << "; ns per keep, today " << shown(4) << ", with the switches " << shown(5) << "\n";
        }
    }
    {
        const auto scene = RepeatScene::Still;
        const std::string name = RepeatSceneName(scene);
        RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, false);
        RuntimeSwitches::Assign(RuntimeSwitch::CommitBindings, false);
        RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsDynamic, false);
        std::vector<std::vector<std::uint8_t>> today;
        for (std::uint32_t round = 0; round < RepeatRounds; ++round) today.push_back(RenderScene(scene, round));
        RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, true);
        const auto run = [&](bool bindings, bool dynamic, const std::string& what) {
            RuntimeSwitches::Assign(RuntimeSwitch::CommitBindings, bindings);
            RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsStamps, bindings);
            RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsSets, false);
            RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsDynamic, dynamic);
            const auto before = DirectTotals::Now();
            for (std::uint32_t round = 0; round < RepeatRounds; ++round) Require(RenderScene(scene, round) == today[round], "commit bindings, " + name + ", " + what + ": round " + std::to_string(round) + " rendered differently from today's path" + DescribeDirect(DirectTotals::Now(), before));
            const auto after = DirectTotals::Now();
            std::cout << "commit bindings, " << name << ", " << what << ": " << after.Since(before, DirectCounter::BindingsDynamicBound) << " dynamic sets bound, " << after.Since(before, DirectCounter::BindingsDynamicWritten) << " written, " << after.Since(before, DirectCounter::BindingsVerifyChecked) << " verified" << DescribeDirect(after, before) << '\n';
            Require(after.Since(before, DirectCounter::BindingsVerifyDiffering) == 0 && after.Since(before, DirectCounter::BindingsRefuted) == 0, "commit bindings, " + name + ", " + what + ": the verify found differences or refuted stamps" + DescribeDirect(after, before));
            Require(PlacedChunksAfterIdle(lingeringChunks()) <= lingeringChunks(), "commit bindings, " + name + ", " + what + ": " + std::to_string(ShaderResources::PlacedUploadChunks()) + " finisher upload chunks are still alive two seconds after the batches completed");
            return std::pair{before, after};
        };
        const auto [plainBefore, plainAfter] = run(true, false, "without dynamic sets");
        const auto [dynamicBefore, dynamicAfter] = run(true, true, "with dynamic sets");
        if (committed) {
            Require(plainAfter.Since(plainBefore, DirectCounter::BindingsTakenWritten) != 0, "commit bindings, " + name + ": no commit wrote a set on today's route" + DescribeDirect(plainAfter, plainBefore));
            const auto bound = dynamicAfter.Since(dynamicBefore, DirectCounter::BindingsDynamicBound);
            const auto written = dynamicAfter.Since(dynamicBefore, DirectCounter::BindingsDynamicWritten);
            Require(bound != 0 && written < bound, "commit bindings, " + name + ": " + std::to_string(bound) + " draws bound dynamic sets, " + std::to_string(written) + " sets were written" + DescribeDirect(dynamicAfter, dynamicBefore));
            if (verifyEvery != 0) Require(dynamicAfter.Since(dynamicBefore, DirectCounter::BindingsVerifyChecked) != 0, "commit bindings, " + name + ": the verify checked no dynamic draw" + DescribeDirect(dynamicAfter, dynamicBefore));
            ShaderResources::SetUploadChunkBytesForTest(SmallChunkBytes);
            const auto [smallBefore, smallAfter] = run(true, true, "with dynamic sets and small finisher chunks");
            ShaderResources::SetUploadChunkBytesForTest(0);
            const auto smallBound = smallAfter.Since(smallBefore, DirectCounter::BindingsDynamicBound);
            const auto smallWritten = smallAfter.Since(smallBefore, DirectCounter::BindingsDynamicWritten);
            Require(smallBound != 0 && smallWritten > written && smallWritten <= smallBound, "commit bindings, " + name + ": with a new finisher chunk every few draws " + std::to_string(smallBound) + " draws bound dynamic sets and " + std::to_string(smallWritten) + " sets were written, against " + std::to_string(written) + " with whole chunks" + DescribeDirect(smallAfter, smallBefore));
            ShaderResources::SetDynamicLimitForTest(0);
            const auto [limitBefore, limitAfter] = run(true, true, "with dynamic sets over the device limit");
            ShaderResources::SetDynamicLimitForTest(std::nullopt);
            Require(limitAfter.Since(limitBefore, DirectCounter::BindingsDynamicBound) == 0 && limitAfter.Since(limitBefore, DirectCounter::BindingsDynamicRefused) != 0, "commit bindings, " + name + ": a template over the dynamic descriptor limit bound a dynamic set or was not counted" + DescribeDirect(limitAfter, limitBefore));
            ShaderResources::SetUploadChunkBytesForTest(SmallChunkBytes);
            RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsDynamic, true);
            const auto fullBefore = ShaderResources::UploadRingTotals();
            Require(RenderScene(scene, 0) == today[0], "commit bindings, " + name + ": a pass that overflows the upload ring rendered differently from today's path");
            const auto fullAfter = ShaderResources::UploadRingTotals();
            Require(RenderScene(scene, 1) == today[1], "commit bindings, " + name + ": the pass after an overflowing one rendered differently from today's path");
            const auto nextAfter = ShaderResources::UploadRingTotals();
            std::cout << "commit bindings, " << name << ", upload ring with " << SmallChunkBytes << "-byte chunks: first pass " << fullAfter.added - fullBefore.added << " chunks added, " << fullAfter.reused - fullBefore.reused << " reused, " << fullAfter.overflow - fullBefore.overflow << " overflow; next pass " << nextAfter.reused - fullAfter.reused << " reused, " << nextAfter.overflow - fullAfter.overflow << " overflow\n";
            Require(fullAfter.reused == fullBefore.reused && fullAfter.overflow > fullBefore.overflow && nextAfter.reused > fullAfter.reused, "commit bindings, " + name + ": the upload ring reused a chunk while its batch was open, did not overflow, or did not reuse its chunks once the batch completed");
            ShaderResources::SetUploadChunkBytesForTest(RingChunkBytes);
            const auto ringBefore = ShaderResources::UploadRingTotals();
            const auto [ringPassBefore, ringPassAfter] = run(true, true, "with dynamic sets and a ring of small chunks");
            const auto ringAfter = ShaderResources::UploadRingTotals();
            ShaderResources::SetUploadChunkBytesForTest(0);
            const auto ringWritten = ringPassAfter.Since(ringPassBefore, DirectCounter::BindingsDynamicWritten);
            const auto ringHits = ringPassAfter.Since(ringPassBefore, DirectCounter::BindingsDynamicHits);
            std::cout << "commit bindings, " << name << ", upload ring with " << RingChunkBytes << "-byte chunks: " << ringAfter.added - ringBefore.added << " chunks added, " << ringAfter.reused - ringBefore.reused << " reused, " << ringWritten << " tuples written, " << ringHits << " tuple hits\n";
            Require(ringAfter.reused > ringBefore.reused && ringHits != 0 && ringWritten <= 2u * (ringAfter.added - ringBefore.added), "commit bindings, " + name + ": a tuple written for a ring chunk was not hit again when the ring came round to it" + DescribeDirect(ringPassAfter, ringPassBefore));
        }
        if (std::getenv("APS5_PIPELINE_STATS") != nullptr && verifyEvery == 0) {
            constexpr std::uint32_t TimingRuns = 7;
            std::array<double, 4> best{};
            best.fill(std::numeric_limits<double>::max());
            const auto timing = [&](bool bindings) {
                RuntimeSwitches::Assign(RuntimeSwitch::CommitBindings, bindings);
                RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsStamps, bindings);
                RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsSets, false);
                RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsDynamic, bindings);
                const auto before = DirectTotals::Now();
                for (std::uint32_t round = 0; round < RepeatTimingRounds; ++round) static_cast<void>(RenderScene(scene, round % RepeatRounds));
                return std::pair{before, DirectTotals::Now()};
            };
            for (std::uint32_t timed = 0; timed < TimingRuns; ++timed) {
                const auto [todayBefore, todayAfter] = timing(false);
                const auto [newBefore, newAfter] = timing(true);
                const auto keep = [&](std::size_t at, double value) {
                    if (value != 0.0) best[at] = std::min(best[at], value);
                };
                keep(0, perDraw(todayAfter, todayBefore, DirectCounter::BindingsTodayWrittenNanoseconds, DirectCounter::BindingsTodayWrittenTimed));
                keep(1, perDraw(newAfter, newBefore, DirectCounter::BindingsNewWrittenNanoseconds, DirectCounter::BindingsNewWrittenTimed));
                keep(2, perDraw(todayAfter, todayBefore, DirectCounter::BindingsTodayPushedNanoseconds, DirectCounter::BindingsTodayPushedTimed));
                keep(3, perDraw(newAfter, newBefore, DirectCounter::BindingsNewPushedNanoseconds, DirectCounter::BindingsNewPushedTimed));
            }
            const auto shown = [&](std::size_t at) { return best[at] == std::numeric_limits<double>::max() ? 0.0 : best[at]; };
            std::cout << "commit bindings timing, " << name << ": ns per binding step (best of " << TimingRuns << " runs), today's route with a written set " << shown(0) << ", with dynamic sets " << shown(1) << " (pushed: today " << shown(2) << ", with the switches " << shown(3) << ")\n";
        }
    }
    RuntimeSwitches::Assign(RuntimeSwitch::CommitBindings, false);
    for (std::uint32_t round = 0; round < RepeatRounds; ++round) static_cast<void>(RenderScene(RepeatScene::Constants, round));
    Require(PlacedChunksAfterIdle(0) == 0, "commit bindings: " + std::to_string(AgcDriver::Graphics::ShaderResources::PlacedUploadChunks()) + " finisher upload chunks are still alive after passes without the switch");
    RuntimeSwitches::Assign(RuntimeSwitch::CommitBindings, true);
    RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsStamps, stampsEnvironment);
    RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsSets, setsEnvironment);
    RuntimeSwitches::Assign(RuntimeSwitch::CommitBindingsDynamic, dynamicEnvironment);
    RuntimeSwitches::Assign(RuntimeSwitch::OwnDrawSets, ownSets);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectPassTargets, passTargets);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, false);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeat, repeatEnvironment);
}

constexpr std::uint32_t FeedDroppedCommits = 4;

struct FeedTotals {
    DirectTotals direct;
    std::uint64_t kept = 0;
    std::uint64_t withdrawn = 0;
    std::uint64_t followerCards = 0;
    std::uint64_t leaderCards = 0;
    std::uint64_t guardCards = 0;
    std::uint64_t withoutCards = 0;
    std::uint64_t aheadSteps = 0;
    std::uint64_t verifyChecked = 0;
    std::uint64_t verifyDiffering = 0;
    std::uint64_t drops = 0;

    static FeedTotals Now() {
        using Count = AgcDriver::Graphics::CommitFeedCount;
        using AgcDriver::Graphics::CommitFeedTotal;
        FeedTotals totals;
        totals.direct = DirectTotals::Now();
        totals.kept = CommitFeedTotal(Count::BlocksKept);
        totals.withdrawn = CommitFeedTotal(Count::BlocksWithdrawn);
        totals.followerCards = CommitFeedTotal(Count::CardFollowerUses);
        totals.leaderCards = CommitFeedTotal(Count::CardLeaderUses);
        totals.guardCards = CommitFeedTotal(Count::CardGuardUses);
        totals.withoutCards = CommitFeedTotal(Count::HitsWithoutCard);
        for (std::size_t level = 0; level < 6; ++level) totals.aheadSteps += CommitFeedTotal(static_cast<Count>(static_cast<std::size_t>(Count::AheadLevel0) + level));
        const auto verify = AgcDriver::Graphics::CommitFeedVerifyTotals();
        totals.verifyChecked = verify[0];
        totals.verifyDiffering = verify[1];
        totals.drops = AgcDriver::DriverDetail::CommitBufferDropsForTest();
        return totals;
    }
};

void CommitFeedMode(bool ring) {
    const bool feedEnvironment = RuntimeSwitches::Set(RuntimeSwitch::CommitFeed);
    const bool repeatEnvironment = RuntimeSwitches::Set(RuntimeSwitch::DirectRepeat);
    const bool passTargets = RuntimeSwitches::Set(RuntimeSwitch::DirectPassTargets);
    const bool census = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    const bool committed = ring && RuntimeSwitches::Set(RuntimeSwitch::CommitFragments) && RuntimeSwitches::Set(RuntimeSwitch::DirectBuffers) && RuntimeSwitches::Set(RuntimeSwitch::DirectInputs) && DirectReport::VerifyEvery() != 1 && DrawRecords::VerifyEvery() != 1 && DirectReport::BufferVerifyEvery() == 0 && DirectReport::InputVerifyEvery() == 0;
    RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, true);
    RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, false);
    RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, false);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeat, false);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectPassTargets, true);
    const auto scene = RepeatScene::Constants;
    const std::string name = std::string("commit feed, ") + RepeatSceneName(scene);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, false);
    std::vector<std::vector<std::uint8_t>> today;
    for (std::uint32_t round = 0; round < RepeatRounds; ++round) today.push_back(RenderScene(scene, round));
    RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, true);
    const auto arm = [&](bool on) {
        const std::string what = name + (on ? ", switch on" : ", switch off");
        RuntimeSwitches::Assign(RuntimeSwitch::CommitFeed, on);
        Require(RenderScene(scene, 0) == today[0], what + ": the first round rendered differently from today's path");
        const auto before = FeedTotals::Now();
        for (std::uint32_t round = 1; round < RepeatRounds; ++round) {
            AgcDriver::DriverDetail::SetCommitBufferDropsForTest(RepeatDraws - FeedDroppedCommits, FeedDroppedCommits);
            const auto image = RenderScene(scene, round);
            AgcDriver::DriverDetail::SetCommitBufferDropsForTest(0, 0);
            Require(image == today[round], what + ": round " + std::to_string(round) + ", whose last commits had no prepared buffers, rendered differently from today's path" + DescribeDirect(DirectTotals::Now(), before.direct));
        }
        Require(RenderScene(scene, 0) == today[0], what + ": the round after the commits without prepared buffers rendered differently from today's path");
        const auto after = FeedTotals::Now();
        std::cout << what << ": " << after.drops - before.drops << " commits without prepared buffers, " << after.kept - before.kept << " fragment blocks kept, " << after.withdrawn - before.withdrawn << " withdrawn, " << after.direct.Since(before.direct, DirectCounter::FragmentsNoBlock) << " hits without a block, " << after.direct.Since(before.direct, DirectCounter::FeedCardsMade) << " zone cards made, " << after.direct.Since(before.direct, DirectCounter::FeedCardsOverCapacity) << " over capacity, " << after.followerCards - before.followerCards << " used by followers, " << after.leaderCards - before.leaderCards << " by leaders, " << after.guardCards - before.guardCards << " for guards, " << after.withoutCards - before.withoutCards << " uses without a card, " << after.aheadSteps - before.aheadSteps << " lookahead steps, verify " << after.verifyChecked - before.verifyChecked << " checked" << DescribeDirect(after.direct, before.direct) << '\n';
        Require(after.verifyDiffering == before.verifyDiffering, what + ": the commit feed verify found a difference");
        Require(after.direct.Since(before.direct, DirectCounter::FragmentsVerifyDiffering) == 0 && after.direct.Since(before.direct, DirectCounter::ExpressVerifyDiffering) == 0, what + ": the fragment or express verify found a difference" + DescribeDirect(after.direct, before.direct));
        return std::pair{before, after};
    };
    const auto [offBefore, offAfter] = arm(false);
    const auto [onBefore, onAfter] = arm(true);
    if (committed) {
        Require(offAfter.drops > offBefore.drops && onAfter.drops > onBefore.drops, name + ": no commit lost its prepared buffers to the test hook");
        const auto offMissing = offAfter.direct.Since(offBefore.direct, DirectCounter::FragmentsNoBlock);
        const auto onMissing = onAfter.direct.Since(onBefore.direct, DirectCounter::FragmentsNoBlock);
        Require(offMissing != 0, name + ": with the switch off no hit found its record's fragment block withdrawn by a commit without prepared buffers" + DescribeDirect(offAfter.direct, offBefore.direct));
        Require(onMissing < offMissing, name + ": with the switch on " + std::to_string(onMissing) + " hits found no fragment block, with it off " + std::to_string(offMissing) + DescribeDirect(onAfter.direct, onBefore.direct));
        Require(onAfter.direct.Since(onBefore.direct, DirectCounter::FeedCardsMade) != 0 && offAfter.direct.Since(offBefore.direct, DirectCounter::FeedCardsMade) == 0, name + ": no zone card was made with the switch on, or one was made with it off");
        if (census) {
            Require(onAfter.kept > onBefore.kept && offAfter.kept == offBefore.kept, name + ": no fragment block was kept with the switch on, or one was kept with it off");
            Require(offAfter.withdrawn > offBefore.withdrawn, name + ": with the switch off no commit without prepared buffers withdrew a fragment block");
        }
        if (AgcDriver::Graphics::CommitFeedVerifyEvery() != 0) Require(onAfter.verifyChecked > onBefore.verifyChecked, name + ": the commit feed verify checked nothing");
    }
    RuntimeSwitches::Assign(RuntimeSwitch::CommitFeed, feedEnvironment);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectRepeat, repeatEnvironment);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectPassTargets, passTargets);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, false);
}

constexpr std::uint32_t PlannedBufferBytes = 256;
constexpr std::uint32_t LargeBufferBytes = 8192;
constexpr std::size_t KeptBufferOffset = 0x1000;
constexpr std::size_t InPlaceBufferOffset = 0x2000;
constexpr std::size_t MovedBufferOffset = 0x3000;
constexpr std::size_t LargeBufferOffset = 0x4000;
constexpr std::uint64_t UnmappedAddress = 0x1000;

// Two read-only guest buffers in one binding: the first at `first`, the second always in place.
ShaderRecompiler::DescriptorBinding PlannedBuffers(const GuestBlock& block, std::uint64_t first, std::uint32_t firstBytes) {
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    binding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    binding.descriptorSet = 0;
    binding.binding = 0;
    binding.count = 2;
    const auto moving = RawDescriptor(first, firstBytes);
    const auto still = RawDescriptor(block.At(InPlaceBufferOffset), PlannedBufferBytes);
    binding.guestDescriptor.assign(moving.begin(), moving.end());
    binding.guestDescriptor.insert(binding.guestDescriptor.end(), still.begin(), still.end());
    binding.bufferWritten = {false, false};
    return binding;
}

ShaderRecompiler::DescriptorBinding PlannedData(std::uint32_t value) {
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    binding.role = ShaderRecompiler::DescriptorRole::ShaderData;
    binding.descriptorSet = 0;
    binding.binding = 5;
    binding.count = 1;
    binding.guestDescriptor.assign(ShaderRecompiler::RuntimeAbi::ShaderDataDwords, 0u);
    binding.guestDescriptor[0] = ShaderRecompiler::RuntimeAbi::Version;
    binding.guestDescriptor[ShaderRecompiler::RuntimeAbi::UserDataDword] = value;
    return binding;
}

// PrepareDirectBuffers against today's MovedReadOnlyBuffers and selectDrawSnapshots (through
// PrepareDrawBindings) on one template and the same stages.
int PreparedBuffersMatch() {
    using AgcDriver::Graphics::CompiledShader;
    using AgcDriver::Graphics::ShaderResources;
    using Refusal = ShaderResources::PreparedBuffers::Refusal;
    using ShaderRecompiler::ShaderStage;
    const auto device = OpenVulkanTestDevice();
    if (!device) return VulkanTestSkipped;
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
    Require(context.recorder != nullptr && context.descriptorCache != nullptr, "prepared buffers: the device has no recorder or descriptor cache");
    Require(ShaderResources::DrawUploadLimit() < LargeBufferBytes, "prepared buffers: the draw upload limit does not leave the large buffer over it");
    RuntimeSwitches::Assign(RuntimeSwitch::MovedLargeUploads, false);
    GuestBlock block;
    std::memset(block.Bytes(KeptBufferOffset), 0x11, PlannedBufferBytes);
    std::memset(block.Bytes(InPlaceBufferOffset), 0x22, PlannedBufferBytes);
    std::memset(block.Bytes(MovedBufferOffset), 0x33, PlannedBufferBytes);
    std::memset(block.Bytes(LargeBufferOffset), 0x44, LargeBufferBytes);
    AgcDriver::GuestMemory::CollectWritesUncached(block.At(0), BlockBytes);
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    auto& recorder = *context.recorder;
    ShaderRecompiler::RecompileResult vertex;
    vertex.bindings = {PlannedBuffers(block, block.At(KeptBufferOffset), PlannedBufferBytes), PlannedData(7)};
    vertex.shaderDataDwords = ShaderRecompiler::RuntimeAbi::ShaderDataDwords;
    const ShaderRecompiler::RecompileResult fragment{};
    const AgcDriver::Graphics::ColorTarget target{};
    const auto stagesOf = [&](const ShaderRecompiler::RecompileResult& program) { return std::array<CompiledShader, 2>{{{ShaderStage::Vertex, &program, 0}, {ShaderStage::Fragment, &fragment, 0}}}; };
    const auto expectRefused = [&](const ShaderResources& owner, const ShaderResources::BufferPlan& plan, const ShaderRecompiler::RecompileResult& program, Refusal reason, bool todayMoves, const std::string& what) {
        const auto stages = stagesOf(program);
        const auto moved = owner.MovedReadOnlyBuffers(stages, recorder, &target);
        Require(moved.has_value() == todayMoves, "prepared buffers: today's check " + std::string(todayMoves ? "refused " : "took ") + what);
        ShaderResources::PreparedBuffers prepared;
        Require(!ShaderResources::PrepareDirectBuffers(plan, stages, &target, prepared) && prepared.refusal == reason, "prepared buffers: " + what + " was prepared, or refused for another reason (" + std::to_string(static_cast<int>(prepared.refusal)) + ")");
    };
    {
        ShaderResources resources(context, vertex, fragment, target, 0, 0);
        const auto reads = resources.InPlaceReads();
        const bool inPlace = std::ranges::any_of(reads, [&](const auto& range) { return range.first <= block.At(InPlaceBufferOffset) && block.At(InPlaceBufferOffset) + PlannedBufferBytes <= range.second; });
        if (!resources.Reusable() || !inPlace) {
            std::cout << "skipped, the template reads the guest block through no host import, so no buffer is read in place\n";
            return VulkanTestSkipped;
        }
        const auto plan = resources.DirectBufferPlan(stagesOf(vertex));
        Require(plan != nullptr, "prepared buffers: a reusable template with guest buffers and shader data has no buffer plan");
        Require(resources.DirectBufferPlan(stagesOf(vertex)) == plan, "prepared buffers: the buffer plan was built again");
        const auto expectSame = [&](const ShaderRecompiler::RecompileResult& program, std::size_t movedCount, std::size_t selectedCount, const std::string& what) {
            const auto stages = stagesOf(program);
            const auto today = resources.MovedReadOnlyBuffers(stages, recorder, &target);
            Require(today.has_value() && today->size() == movedCount, "prepared buffers, " + what + ": today's check did not move " + std::to_string(movedCount) + " buffers");
            std::vector<std::size_t> selected;
            const auto bindings = resources.PrepareDrawBindings(recorder, *today, false, nullptr, false, &selected);
            Require(bindings != nullptr && bindings->snapshots.size() == selected.size() && selected.size() == selectedCount, "prepared buffers, " + what + ": today's selection did not snapshot " + std::to_string(selectedCount) + " allocations");
            ShaderResources::PreparedBuffers prepared;
            Require(ShaderResources::PrepareDirectBuffers(*plan, stages, &target, prepared) && prepared.refusal == Refusal::None, "prepared buffers, " + what + ": the draw's buffers were not prepared (" + std::to_string(static_cast<int>(prepared.refusal)) + ")");
            Require(prepared.moved.size() == today->size(), "prepared buffers, " + what + ": the moved lists differ in length");
            for (std::size_t index = 0; index < today->size(); ++index) {
                const auto& mine = prepared.moved[index];
                const auto& theirs = (*today)[index];
                Require(mine.allocation == theirs.allocation && mine.address == theirs.address && mine.size == theirs.size && std::ranges::equal(mine.words, theirs.words), "prepared buffers, " + what + ": moved entry " + std::to_string(index) + " differs");
            }
            Require(prepared.selected.size() == selected.size(), "prepared buffers, " + what + ": the selections differ in length");
            std::array<std::size_t, 3> kinds{};
            for (std::size_t index = 0; index < selected.size(); ++index) {
                const auto& mine = prepared.selected[index];
                const auto& snapshot = bindings->snapshots[index];
                Require(mine.allocation == selected[index] && mine.begin == snapshot.address && mine.bytes == snapshot.bytes, "prepared buffers, " + what + ": selected allocation " + std::to_string(index) + " has another index or guest range");
                Require(mine.offset % plan->alignment == 0 && mine.offset + mine.bytes <= prepared.blob.size(), "prepared buffers, " + what + ": selected allocation " + std::to_string(index) + " has an unaligned or outlying slice");
                const auto bytes = snapshot.Bytes();
                Require(bytes.size() == mine.bytes && std::memcmp(bytes.data(), prepared.blob.data() + mine.offset, mine.bytes) == 0, "prepared buffers, " + what + ": selected allocation " + std::to_string(index) + " holds other bytes");
                ++kinds[!mine.moved ? 0 : mine.begin == 0 ? 1 : 2];
            }
            return kinds;
        };
        auto drawn = vertex;
        drawn.bindings[0] = PlannedBuffers(block, block.At(MovedBufferOffset), PlannedBufferBytes);
        drawn.bindings[1] = PlannedData(9);
        const auto kinds = expectSame(drawn, 2, 3, "a moved buffer, an in-place buffer and changed shader data");
        Require(kinds == std::array<std::size_t, 3>{1, 1, 1}, "prepared buffers: the draw did not select one in-place buffer, one moved data block and one moved guest buffer");
        const auto still = expectSame(vertex, 0, 2, "the template's own stages");
        Require(still == std::array<std::size_t, 3>{2, 0, 0}, "prepared buffers: the template's own draw did not select its two in-place buffers");
        // The committer's half: the moved-range check leaves the ticket as today's does (the second
        // round trusts the ranges the first one stamped), the verify finds the two ways equal and a
        // changed byte or moved size, and the prepared selection binds today's bytes from one slice.
        const auto expectCommitted = [&](const ShaderRecompiler::RecompileResult& program, std::size_t movedRanges, const std::string& what) {
            const auto stages = stagesOf(program);
            ShaderResources::PreparedBuffers prepared;
            Require(ShaderResources::PrepareDirectBuffers(*plan, stages, &target, prepared), "prepared buffers, " + what + ": the draw's buffers were not prepared");
            prepared.plan = plan;
            std::vector<std::pair<std::uint64_t, std::uint64_t>> todayRanges;
            std::vector<std::pair<std::uint64_t, std::uint64_t>> preparedRanges;
            ShaderResources::MoveTicket todayTicket;
            ShaderResources::MoveTicket preparedTicket;
            todayTicket.ranges = &todayRanges;
            preparedTicket.ranges = &preparedRanges;
            std::optional<std::vector<ShaderResources::MovedBuffer>> today;
            for (std::size_t round = 0; round < 2; ++round) {
                todayTicket.checked = todayTicket.trusted = preparedTicket.checked = preparedTicket.trusted = 0;
                today = resources.MovedReadOnlyBuffers(stages, recorder, &target, nullptr, &todayTicket);
                Require(today.has_value() && resources.PreparedMovesClear(prepared, recorder, &preparedTicket), "prepared buffers, " + what + ": a moved range was refused");
                const bool same = todayTicket.valid == preparedTicket.valid && todayTicket.recorder == preparedTicket.recorder && todayTicket.note == preparedTicket.note && todayTicket.pending == preparedTicket.pending && todayTicket.allocations == preparedTicket.allocations && todayTicket.epoch == preparedTicket.epoch && todayTicket.checked == preparedTicket.checked && todayTicket.trusted == preparedTicket.trusted && todayRanges == preparedRanges;
                Require(same && preparedTicket.checked == movedRanges && preparedTicket.trusted == (round == 0 ? 0 : movedRanges), "prepared buffers, " + what + ": round " + std::to_string(round) + " of the prepared moved-range check left the ticket otherwise than today's check");
                Require(ShaderResources::ComparePreparedTickets(&preparedTicket, todayTicket).what == nullptr, "prepared buffers, " + what + ": round " + std::to_string(round) + ": the verify found the tickets different");
            }
            auto otherTicket = preparedTicket;
            ++otherTicket.checked;
            Require(ShaderResources::ComparePreparedTickets(&otherTicket, todayTicket).what != nullptr && ShaderResources::ComparePreparedTickets(nullptr, todayTicket).what != nullptr, "prepared buffers, " + what + ": the verify did not find a ticket of another count, or a refused prepared check");
            Require(ShaderResources::ComparePreparedMoves(prepared, *today).what == nullptr, "prepared buffers, " + what + ": the verify found the moved lists different");
            std::vector<std::size_t> selected;
            const auto bindings = resources.PrepareDrawBindings(recorder, *today, false, nullptr, false, &selected);
            Require(ShaderResources::ComparePreparedSelection(prepared, bindings.get(), selected).what == nullptr, "prepared buffers, " + what + ": the verify found the selections different");
            auto corrupted = prepared;
            corrupted.blob[corrupted.selected.back().offset] ^= std::byte{0xff};
            const auto changed = ShaderResources::ComparePreparedSelection(corrupted, bindings.get(), selected);
            Require(changed.what != nullptr && std::string_view(changed.what) == "bytes" && !changed.memoryChanged, "prepared buffers, " + what + ": the verify did not find a changed byte");
            if (movedRanges != 0) DirectReport::NoteBufferDifference(changed.what, changed.detail);
            if (!prepared.moved.empty()) {
                auto resized = prepared;
                resized.moved.front().size += 4;
                Require(ShaderResources::ComparePreparedMoves(resized, *today).what != nullptr, "prepared buffers, " + what + ": the verify did not find a moved buffer of another size");
            }
            Require(resources.ComparePreparedBinding(recorder, *today, prepared, bindings.get(), selected).what == nullptr, "prepared buffers, " + what + ": the verify found the bound selection different from today's");
            if (!selected.empty()) {
                const std::vector<std::size_t> fewer(selected.begin(), selected.end() - 1);
                const auto shorter = resources.ComparePreparedBinding(recorder, *today, prepared, bindings.get(), fewer);
                Require(shorter.what != nullptr && std::string_view(shorter.what) == "bound selection", "prepared buffers, " + what + ": the verify did not find a bound selection of another length");
            }
            std::vector<std::size_t> taken;
            auto used = ShaderResources::PreparedBuffers::Selection::Hazard;
            const auto bound = resources.PrepareDrawBindings(recorder, prepared.moved, false, nullptr, false, &taken, &prepared, &used);
            Require(used == ShaderResources::PreparedBuffers::Selection::Taken && bound != nullptr && taken == selected && bound->snapshots.size() == bindings->snapshots.size(), "prepared buffers, " + what + ": the prepared selection was not bound, or bound other allocations");
            const auto alignment = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 1);
            for (std::size_t index = 0; index < taken.size(); ++index) {
                const auto& mine = bound->snapshots[index];
                const auto& theirs = bindings->snapshots[index];
                Require(mine.buffer == bound->snapshots.front().buffer && mine.offset % alignment == 0, "prepared buffers, " + what + ": snapshot " + std::to_string(index) + " is not an aligned part of one upload slice");
                Require(mine.address == theirs.address && mine.bytes == theirs.bytes && std::memcmp(mine.Bytes().data(), theirs.Bytes().data(), mine.bytes) == 0, "prepared buffers, " + what + ": snapshot " + std::to_string(index) + " binds another range or other bytes than today's");
            }
        };
        expectCommitted(drawn, 1, "a moved buffer, an in-place buffer and changed shader data, on the committer");
        expectCommitted(vertex, 0, "the template's own stages, on the committer");
        auto longer = vertex;
        longer.bindings[1].guestDescriptor.push_back(0u);
        expectRefused(resources, *plan, longer, Refusal::Classification, false, "shader data of another size");
        auto unmapped = vertex;
        unmapped.bindings[0] = PlannedBuffers(block, UnmappedAddress, PlannedBufferBytes);
        expectRefused(resources, *plan, unmapped, Refusal::Inaccessible, false, "a buffer moved to unmapped memory");
        auto large = vertex;
        large.bindings[0] = PlannedBuffers(block, block.At(LargeBufferOffset), LargeBufferBytes);
        expectRefused(resources, *plan, large, Refusal::UploadLimit, true, "a buffer moved to a range over the upload limit");
    }
    SetTestEnvironment("APS5_DRAW_SNAPSHOT_MAX_MIB", "0");
    {
        ShaderResources capped(context, vertex, fragment, target, 0, 0);
        SetTestEnvironment("APS5_DRAW_SNAPSHOT_MAX_MIB", "64");
        const auto plan = capped.DirectBufferPlan(stagesOf(vertex));
        Require(plan != nullptr, "prepared buffers: a template with no snapshot budget has no buffer plan");
        auto moving = vertex;
        moving.bindings[0] = PlannedBuffers(block, block.At(MovedBufferOffset), PlannedBufferBytes);
        expectRefused(capped, *plan, moving, Refusal::Budget, false, "a moved buffer over the snapshot budget");
    }
    // Last, as it leaves pending writes in the open batch: a guest write after the finisher's read is
    // told apart as memory changed, and pending writes send the committer to today's way.
    {
        ShaderResources resources(context, vertex, fragment, target, 0, 0);
        const auto plan = resources.DirectBufferPlan(stagesOf(vertex));
        auto drawn = vertex;
        drawn.bindings[0] = PlannedBuffers(block, block.At(MovedBufferOffset), PlannedBufferBytes);
        const auto stages = stagesOf(drawn);
        ShaderResources::PreparedBuffers prepared;
        Require(plan != nullptr && ShaderResources::PrepareDirectBuffers(*plan, stages, &target, prepared), "prepared buffers, hazards: the draw's buffers were not prepared");
        prepared.plan = plan;
        const auto inPlace = std::ranges::find_if(prepared.selected, [](const ShaderResources::PreparedBuffers::Selected& item) { return !item.moved; });
        Require(inPlace != prepared.selected.end(), "prepared buffers, hazards: no in-place buffer was selected");
        // A moved buffer in memory no template imports, which the write tracker watches.
        GuestBlock outside;
        std::memset(outside.Bytes(0), 0x66, PlannedBufferBytes);
        auto away = vertex;
        away.bindings[0] = PlannedBuffers(block, outside.At(0), PlannedBufferBytes);
        const auto awayStages = stagesOf(away);
        ShaderResources::PreparedBuffers early;
        Require(ShaderResources::PrepareDirectBuffers(*plan, awayStages, &target, early), "prepared buffers, hazards: a buffer moved to another block was not prepared");
        early.plan = plan;
        const bool watched = AgcDriver::GuestMemory::Watched(outside.At(0), PlannedBufferBytes);
        std::cout << (watched ? "a guest write after the finisher's read is checked to count as memory changed\n" : "the moved buffer's block is not write-watched, so a guest write after the finisher's read is not checked to count as memory changed\n");
        if (watched) {
            early.generation = AgcDriver::GuestMemory::TrackerGeneration();
            std::memset(outside.Bytes(0), 0x77, PlannedBufferBytes);
            AgcDriver::GuestMemory::CollectWritesUncached(outside.At(0), BlockBytes);
            const auto today = resources.MovedReadOnlyBuffers(awayStages, recorder, &target);
            Require(today.has_value(), "prepared buffers, hazards: today's check refused a buffer moved to another block");
            std::vector<std::size_t> selected;
            const auto bindings = resources.PrepareDrawBindings(recorder, *today, false, nullptr, false, &selected);
            const auto difference = ShaderResources::ComparePreparedSelection(early, bindings.get(), selected);
            Require(difference.what != nullptr && difference.memoryChanged, "prepared buffers, hazards: a guest write between the finisher's read and today's was not told apart as memory changed (" + std::string(difference.what != nullptr ? difference.what : "no difference") + ")");
        }
        // A mapping change since the finisher found the moved range accessible makes the committer
        // test it again.
        auto gone = prepared;
        const auto movedEntry = std::ranges::find_if(gone.moved, [](const ShaderResources::MovedBuffer& entry) { return entry.address != 0; });
        Require(movedEntry != gone.moved.end(), "prepared buffers, hazards: no moved guest buffer was prepared");
        movedEntry->address = UnmappedAddress;
        gone.forget = AgcDriver::GuestMemory::ForgetSerial();
        Require(resources.PreparedMovesClear(gone, recorder, nullptr), "prepared buffers, hazards: a moved range the finisher found accessible was refused while no mapping changed");
        gone.forget += 2;
        Require(!resources.PreparedMovesClear(gone, recorder, nullptr), "prepared buffers, hazards: an unmapped moved range was taken after a mapping changed");
        // A depth guard set up or dropped since the finisher's read sends the selection to today's way.
        using Selection = ShaderResources::PreparedBuffers::Selection;
        auto guarded = prepared;
        ++guarded.depthGuards;
        std::vector<std::size_t> selected;
        auto used = Selection::Taken;
        static_cast<void>(resources.PrepareDrawBindings(recorder, guarded.moved, false, nullptr, false, &selected, &guarded, &used));
        Require(used == Selection::DepthGuards, "prepared buffers, hazards: a depth guard change since the finisher's read did not send the selection to today's way");
        recorder.NotePendingWrite(block.At(InPlaceBufferOffset), PlannedBufferBytes);
        Require(resources.PreparedMovesClear(prepared, recorder, nullptr), "prepared buffers, hazards: a pending write beside the moved range refused it");
        used = Selection::Taken;
        static_cast<void>(resources.PrepareDrawBindings(recorder, prepared.moved, false, nullptr, false, &selected, &prepared, &used));
        Require(used == Selection::Hazard && std::ranges::find(selected, inPlace->allocation) == selected.end(), "prepared buffers, hazards: a pending write over the in-place range did not send the selection to today's way, which leaves the range out");
        recorder.NotePendingWrite(block.At(MovedBufferOffset), PlannedBufferBytes);
        Require(!resources.PreparedMovesClear(prepared, recorder, nullptr), "prepared buffers, hazards: a pending write over the moved range did not refuse it");
        recorder.Sync();
    }
    std::cout << "prepared buffers match today's moved buffers and snapshot selection for a moved buffer, an in-place buffer and changed shader data, and are refused for shader data of another size, unmapped memory, a range over the upload limit and the snapshot budget; on the committer they leave the ticket as today's check does, bind today's bytes from one aligned upload slice, and give way to today's check and selection under pending writes, a depth guard change since the read, and an unmapped moved range after a mapping change; the verify tells a changed byte, a resized moved buffer, another ticket, another bound selection and a guest write after the read apart\n";
    return 0;
}

constexpr std::size_t InputIndexOffset = 0x100;
constexpr std::size_t InputVertexOffset = 0x400;
constexpr std::uint32_t InputVertices = 4;

ShaderRecompiler::VertexAttribute InputAttribute(std::uint64_t address, std::uint32_t stride, std::uint32_t records) {
    return ShaderRecompiler::VertexAttribute{0, 4, {{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) | (stride << 16u), records, 77u << 12u}}, 0};
}

int PreparedInputsMatch() {
    using AgcDriver::DrawRecordInputs;
    using AgcDriver::Graphics::CompiledShader;
    using AgcDriver::Graphics::PreparedInputs;
    using Refusal = PreparedInputs::Refusal;
    using ShaderRecompiler::ShaderStage;
    namespace Graphics = AgcDriver::Graphics;
    const auto device = OpenVulkanTestDevice();
    if (!device) return VulkanTestSkipped;
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
    Require(context.recorder != nullptr, "prepared inputs: the device has no recorder");
    auto& recorder = *context.recorder;
    GuestBlock block;
    const std::array<std::uint16_t, 6> indices{0, 1, 2, 2, 1, 3};
    std::memcpy(block.Bytes(InputIndexOffset), indices.data(), sizeof(indices));
    for (std::uint32_t at = 0; at < InputVertices; ++at) {
        const std::array<float, 4> value{static_cast<float>(at), 0.0f, 0.0f, 1.0f};
        std::memcpy(block.Bytes(InputVertexOffset + at * 16u), value.data(), sizeof(value));
    }
    AgcDriver::GuestMemory::CollectWritesUncached(block.At(0), BlockBytes);
    ShaderRecompiler::RecompileResult vertex;
    vertex.vertexAttributes = {InputAttribute(block.At(InputVertexOffset), 16, InputVertices)};
    const ShaderRecompiler::RecompileResult fragment{};
    const std::array<CompiledShader, 2> stages{{{ShaderStage::Vertex, &vertex, 0}, {ShaderStage::Fragment, &fragment, 0}}};
    Graphics::State state{};
    state.stages.path = Graphics::ShaderPath::Vertex;
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, 64, 64, -64, 0, 1};
    state.scissor = {{0, 0}, {64, 64}};
    state.renderExtent = {64, 64};
    auto recipe = std::make_shared<AgcDriver::DrawRecipe>();
    recipe->device = context.device;
    recipe->vertexInput = Graphics::BuildVertexInputLayout(context, vertex.vertexAttributes);
    recipe->fragmentOutputs = 1;
    recipe->shaderStages = Graphics::PipelineStages(stages);
    const AgcDriver::Pm4::DrawParameters indexed{block.At(InputIndexOffset), static_cast<std::uint32_t>(indices.size()), 2, 1, 0, true, 0, 0};
    const AgcDriver::Pm4::DrawParameters automatic{0, InputVertices, 0, 1, 0, false, 0, 0};
    const auto prepare = [&](const AgcDriver::Pm4::DrawParameters& draw, PreparedInputs& prepared, const std::string& what) {
        Require(Graphics::PrepareDrawInputs(context, state, draw, stages, recipe, prepared), "prepared inputs, " + what + ": the draw's inputs were not prepared (" + std::to_string(static_cast<int>(prepared.refusal)) + ")");
    };
    const auto verify = [&](const Graphics::State& at, const AgcDriver::Pm4::DrawParameters& draw, PreparedInputs& prepared, DrawRecordInputs wanted, const std::string& what, DrawRecordInputs took = DrawRecordInputs::None) {
        std::string difference;
        auto taken = DrawRecordInputs::None;
        const auto use = Graphics::VerifyPreparedInputsForTest(context, at, draw, stages, *recipe, prepared, difference, &taken);
        Require(use == wanted, "prepared inputs, " + what + ": the committer answered " + std::to_string(static_cast<int>(use)) + " instead of " + std::to_string(static_cast<int>(wanted)) + (difference.empty() ? std::string() : " (" + difference + ")"));
        Require(took == DrawRecordInputs::None || taken == took, "prepared inputs, " + what + ": the committer took the inputs as " + std::to_string(static_cast<int>(taken)) + " instead of " + std::to_string(static_cast<int>(took)));
        return difference;
    };
    const bool memoOff = RuntimeSwitches::Set(RuntimeSwitch::NoInputMemo);
    RuntimeSwitches::Assign(RuntimeSwitch::NoInputMemo, true);
    struct MemoRestore {
        bool off;
        ~MemoRestore() { RuntimeSwitches::Assign(RuntimeSwitch::NoInputMemo, off); }
    } memoRestore{memoOff};
    const auto refused = [&](const Graphics::State& at, const AgcDriver::Pm4::DrawParameters& draw, std::shared_ptr<const AgcDriver::DrawRecipe> with, Refusal reason, const std::string& what) {
        PreparedInputs prepared;
        Require(!Graphics::PrepareDrawInputs(context, at, draw, stages, std::move(with), prepared) && prepared.refusal == reason, "prepared inputs: " + what + " was prepared, or refused for another reason (" + std::to_string(static_cast<int>(prepared.refusal)) + ")");
    };
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        if (!context.drawInputsInPlace || Graphics::HostImportFor(context, block.At(0), BlockBytes) == nullptr) {
            std::cout << "skipped, the device binds no draw input in place from a host import\n";
            return VulkanTestSkipped;
        }
        PreparedInputs prepared;
        prepare(indexed, prepared, "an indexed draw");
        Require(prepared.ranges.size() == 2 && prepared.maxIndex == 3 && prepared.inPlaceReads.size() == 2 && prepared.vertexHandles.size() == 1 && prepared.vertexOffsets.size() == 1, "prepared inputs: the indexed draw did not prepare one index range and one vertex range up to vertex 3");
        Require(prepared.inPlaceReads.front() == std::pair<std::uint64_t, std::uint64_t>{block.At(InputIndexOffset), block.At(InputIndexOffset) + sizeof(indices)}, "prepared inputs: the index range is not the draw's first in-place read");
        verify(state, indexed, prepared, DrawRecordInputs::Checked, "an indexed in-place draw", DrawRecordInputs::Used);
        PreparedInputs drawn;
        prepare(automatic, drawn, "an auto draw");
        Require(drawn.ranges.size() == 1 && drawn.maxIndex == InputVertices - 1u, "prepared inputs: the auto draw did not prepare one vertex range up to its last vertex");
        verify(state, automatic, drawn, DrawRecordInputs::Checked, "an auto draw", DrawRecordInputs::Used);
        PreparedInputs elsewhere;
        prepare(indexed, elsewhere, "an indexed draw for another state");
        auto copy = state;
        verify(copy, indexed, elsewhere, DrawRecordInputs::Shape, "the inputs prepared for another state object");
        PreparedInputs corrupted;
        prepare(indexed, corrupted, "an indexed draw to corrupt");
        ++corrupted.maxIndex;
        const auto difference = verify(state, indexed, corrupted, DrawRecordInputs::Differs, "a corrupted max index");
        Require(difference.find("max index") != std::string::npos, "prepared inputs: the verify did not name the max index: " + difference);
        PreparedInputs offset;
        prepare(indexed, offset, "an indexed draw to move");
        offset.vertexOffsets.front() += 16;
        Require(verify(state, indexed, offset, DrawRecordInputs::Differs, "a moved vertex offset").find("vertex binding 0") != std::string::npos, "prepared inputs: the verify did not name the vertex binding");
        auto quads = state;
        quads.quadList = true;
        refused(quads, automatic, recipe, Refusal::Shape, "a quad list");
        refused(state, indexed, nullptr, Refusal::NoRecipe, "a draw without a published recipe");
        auto foreign = std::make_shared<AgcDriver::DrawRecipe>(*recipe);
        foreign->device = VK_NULL_HANDLE;
        refused(state, indexed, foreign, Refusal::NoRecipe, "a draw with a recipe of another device");
        if (!context.primitiveListRestart) {
            auto restart = state;
            restart.primitiveRestart = true;
            refused(restart, indexed, recipe, Refusal::Shape, "an indexed list with primitive restart compacted on the CPU");
        }
        alignas(16) static std::array<std::uint16_t, 6> loose{0, 1, 2, 2, 1, 3};
        const AgcDriver::Pm4::DrawParameters unimported{reinterpret_cast<std::uintptr_t>(loose.data()), static_cast<std::uint32_t>(loose.size()), 2, 1, 0, true, 0, 0};
        refused(state, unimported, recipe, Refusal::NotInPlace, "an index buffer in memory without a host import");
        const auto kept = vertex.vertexAttributes;
        vertex.vertexAttributes = {InputAttribute(block.At(InputVertexOffset), 0, 0)};
        refused(state, automatic, recipe, Refusal::Shape, "a vertex fetch that reads zeros");
        vertex.vertexAttributes = kept;
        PreparedInputs rechecked;
        prepare(indexed, rechecked, "an indexed draw before another import");
        const auto* before = Graphics::HostImportFor(context, block.At(0), BlockBytes);
        Require(before != nullptr, "prepared inputs: the first block has no import");
        const auto beforeBuffer = before->buffer;
        const auto beforeOffset = Graphics::HostImportOffset(*before, block.At(0));
        GuestBlock other;
        Require(Graphics::HostImportFor(context, other.At(0), BlockBytes) != nullptr, "prepared inputs: the second block was not imported");
        Require(Graphics::HostImportSetGeneration() != rechecked.imports, "prepared inputs: a new import did not move the import set generation");
        const auto* after = Graphics::HostImportFor(context, block.At(0), BlockBytes);
        const bool replaced = after == nullptr || after->buffer != beforeBuffer || Graphics::HostImportOffset(*after, block.At(0)) != beforeOffset;
        if (replaced) std::cout << "the second block's import replaced the first block's, so the committer drops the prepared inputs\n";
        verify(state, indexed, rechecked, replaced ? DrawRecordInputs::ImportChanged : DrawRecordInputs::Checked, "an indexed draw after another block's import", replaced ? DrawRecordInputs::ImportChanged : DrawRecordInputs::Rechecked);
    }
    PreparedInputs moved;
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        prepare(indexed, moved, "an indexed draw before its block is registered again");
    }
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block.Bytes(0));
    }
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block.Bytes(0), BlockBytes, true, true, true);
    }
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        const auto* import = Graphics::HostImportFor(context, block.At(0), BlockBytes);
        Require(import != nullptr, "prepared inputs: the block registered again was not imported");
        Require(Graphics::HostImportSetGeneration() != moved.imports, "prepared inputs: registering the block again did not move the import set generation");
        const bool reused = import->buffer == moved.ranges.front().handle && Graphics::HostImportOffset(*import, block.At(InputIndexOffset)) == moved.ranges.front().offset;
        if (reused) std::cout << "the import made after the block was registered again has the old import's buffer handle and offset, so the committer takes the prepared inputs\n";
        verify(state, indexed, moved, reused ? DrawRecordInputs::Checked : DrawRecordInputs::ImportChanged, "an indexed draw after its block was registered again");
        recorder.NotePendingWrite(block.At(InputVertexOffset), 16);
        PreparedInputs behind;
        prepare(indexed, behind, "an indexed draw behind a pending vertex write");
        verify(state, indexed, behind, DrawRecordInputs::Checked, "an indexed draw behind a pending vertex write");
        PreparedInputs stored;
        prepare(indexed, stored, "an indexed draw under a queued store");
        const std::array<std::byte, 4> label{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
        recorder.RecordStore(import->buffer, Graphics::HostImportOffset(*import, block.At(InputIndexOffset)), label, block.At(InputIndexOffset));
        Require(recorder.QueuedStoreOverlaps(block.At(InputIndexOffset), label.size()), "prepared inputs: the label store over the index range was not queued");
        verify(state, indexed, stored, DrawRecordInputs::Refused, "an indexed draw under a queued store");
        recorder.Sync();
    }
    std::cout << "prepared inputs match today's for an indexed and an auto in-place draw, also after another block's import moved the import set; they are refused for a quad list, a missing or foreign recipe, CPU-compacted restart lists, an index buffer without a host import and a zero-reading vertex fetch; on the committer they give way to today's inputs for another state object, an import replaced by a registry change and a queued store over the index range, and keep the behind-a-write answer of today's check; the verify names a changed max index and vertex offset\n";
    return 0;
}

}

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string_view(argv[1]) == "--direct-buffers") return PreparedBuffersMatch();
        if (argc > 1 && std::string_view(argv[1]) == "--direct-inputs") return PreparedInputsMatch();
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        const bool inject = argc > 1 && std::string_view(argv[1]) == "--inject";
        const bool injectCommit = argc > 1 && std::string_view(argv[1]) == "--inject-commit";
        const bool injectPlans = argc > 1 && std::string_view(argv[1]) == "--inject-plans";
        const bool texturesOnly = argc > 1 && std::string_view(argv[1]) == "--textures";
        const bool commitFeed = argc > 1 && std::string_view(argv[1]) == "--commit-feed";
        using AgcDriver::DriverDetail::DrawFinisher;
        using AgcDriver::DriverDetail::DrawPipeline;
        const bool leanEnvironment = RuntimeSwitches::Set(RuntimeSwitch::LeanFront);
        const bool recordsEnvironment = RuntimeSwitches::Set(RuntimeSwitch::DrawRecords);
        const bool plansEnvironment = RuntimeSwitches::Set(RuntimeSwitch::StagePlans);
        const bool directEnvironment = RuntimeSwitches::Set(RuntimeSwitch::DirectDraws);
        RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, false);
        Arms arms;
        arms.ring = DrawFinisher::Enabled() && DrawFinisher::Overlap() && DrawPipeline::Depth() != 0;
        PlanArms planArms;
        planArms.ring = arms.ring;
        Register(PlainVertexHeader, PlainVertex.data(), sizeof(PlainVertex), 2);
        Register(ConstantPixelHeader, ConstantPixel.data(), sizeof(ConstantPixel), 1);
        Register(TableVertexHeader, TableVertex.data(), sizeof(TableVertex), 2);
        Register(ReadPixelHeader, ReadPixel.data(), sizeof(ReadPixel), 1);
        Register(RingPixelHeader, RingPixel.data(), sizeof(RingPixel), 1);
        Register(SamplePixelHeader, SamplePixel.data(), sizeof(SamplePixel), 1);
        Register(DataVertexHeader, DataVertex.data(), sizeof(DataVertex), 2);
        RegisterFetchVertex();
        Blocks blocks;
        Scene = &blocks;
        Prepare();
        if (commitFeed) {
            CommitFeedMode(arms.ring);
            AgcDriver::DriverDetail::DirectDraws::Report(true);
            RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, leanEnvironment);
            RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, recordsEnvironment);
            RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, plansEnvironment);
            RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, directEnvironment);
            AgcDriverShutdown_nid_postfix();
            std::cout << "commit feed: records whose commits lost their prepared buffers render as today's path in both arms; with the switch on their fragment blocks stay and zone cards are made" << (arms.ring ? "" : " (no finisher ring: records never ran)") << '\n';
            return 0;
        }
        if (directEnvironment) Direct(arms.ring);
        if (directEnvironment && RuntimeSwitches::Set(RuntimeSwitch::DirectRepeat)) Repeats(arms.ring);
        if (directEnvironment && RuntimeSwitches::Set(RuntimeSwitch::CommitBindings)) CommitBindings(arms.ring);
        if (directEnvironment && RuntimeSwitches::Set(RuntimeSwitch::WriteValidation)) WriteValidationEvents(arms.ring);
        if (inject) {
            Injected(arms);
        } else if (injectCommit) {
            InjectedCommit(arms);
        } else if (texturesOnly) {
            Textures(planArms);
        } else if (injectPlans) {
            InjectedPlans(planArms);
        } else if (plansEnvironment) {
            Plans(planArms);
            Textures(planArms);
        } else {
            Hits(arms);
            Ring(arms);
            Formats(arms);
            Misses(arms);
            Fallbacks(arms);
            Plans(planArms);
            Textures(planArms);
        }
        DrawRecords::Report(true);
        StagePlanReport::Report(true);
        AgcDriver::DriverDetail::DirectDraws::Report(true);
        RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, leanEnvironment);
        RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, recordsEnvironment);
        RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, plansEnvironment);
        RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, directEnvironment);
        AgcDriverShutdown_nid_postfix();
        std::cout << (inject || injectCommit || injectPlans ? "the draw record verify caught a corrupted answer and today's path drew" : "draw record hits render as today's path for moved constants, descriptors and vertex buffers, relocate inline buffer descriptors read through and bound, patch read-only buffer bases and sizes moved inside captured runs, pick the variant of a changed format word, record through the committer half with pushed constants, held targets and per-batch keeps, and fall back on structural changes, writes after the fetch and failed committer checks") << (arms.ring ? "" : " (no finisher ring: records never ran)") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { AgcDriverShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
