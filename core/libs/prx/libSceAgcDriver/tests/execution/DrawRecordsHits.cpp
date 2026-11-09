#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRecords.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgc/Shader/include/ShaderConstants.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

struct Header {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(8) Header PlainVertexHeader{};
alignas(8) Header ConstantPixelHeader{};
alignas(8) Header TableVertexHeader{};
alignas(8) Header ReadPixelHeader{};
alignas(8) Header RingPixelHeader{};

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

using RegisterWrites = std::vector<std::pair<std::uint32_t, std::uint32_t>>;

void Append(std::vector<std::uint32_t>& commands, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    commands.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    commands.insert(commands.end(), payload);
}

void SubmitDraws(std::uint64_t vertex, std::uint64_t pixel, const RegisterWrites& userConfig, const RegisterWrites& shaderRegisters, const GuestBlock& target, std::uint32_t draws) {
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
        {0x3b8, 0x9000000u | (RenderTargetTileMode << 14u)}, {0x1e0, 0},
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
    for (std::uint32_t draw = 0; draw < draws; ++draw) Append(commands, 0x2d, {3, 2});
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

std::array<std::uint32_t, 4> SlotDescriptor(std::uint64_t address) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (4u << 16u), SlotRecords, 0x01014facu};
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
};

Blocks* Scene = nullptr;

enum class Kind { Constants, Table, Fetch, Ring, Inner, RingInner, Sized, Format };

struct State {
    Kind kind = Kind::Table;
    std::uint32_t variant = 0;
    std::uint32_t stride = 16;
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
    default:
        break;
    }
}

void Submit(const State& state, std::uint32_t draws) {
    auto& blocks = *Scene;
    switch (state.kind) {
    case Kind::Constants: {
        const auto pointer = blocks.data.At(ConstantsOffset + state.variant * 0x40u);
        SubmitDraws(reinterpret_cast<std::uintptr_t>(PlainVertex.data()), reinterpret_cast<std::uintptr_t>(ConstantPixel.data()), {{0x242, 4}}, {{0x8b, 0}, {0x8c, 0}, {0x00b, 3u << 1u}, {0x00c, static_cast<std::uint32_t>(pointer)}, {0x00d, static_cast<std::uint32_t>(pointer >> 32u)}, {0x00e, std::bit_cast<std::uint32_t>(0.25f)}}, blocks.target, draws);
        break;
    }
    case Kind::Table: {
        const auto table = blocks.vertices.At(TableOffset);
        const auto pointer = blocks.data.At(0);
        SubmitDraws(reinterpret_cast<std::uintptr_t>(TableVertex.data()), reinterpret_cast<std::uintptr_t>(ReadPixel.data()), {{0x242, 4}}, {{0x8b, 2u << 1u}, {0x8c, static_cast<std::uint32_t>(table)}, {0x8d, static_cast<std::uint32_t>(table >> 32u)}, {0x00b, 8u << 1u}, {0x00c, 0}, {0x00d, 0}, {0x00e, static_cast<std::uint32_t>(pointer)}, {0x00f, static_cast<std::uint32_t>(pointer >> 32u)}}, blocks.target, draws);
        break;
    }
    case Kind::Fetch: {
        const auto stored = OutputDescriptor(blocks.output.At(0));
        const auto buffers = blocks.fetch.At(FetchBufferTableOffset);
        const auto attributes = blocks.fetch.At(FetchAttribTableOffset);
        const std::array<std::uint32_t, 10> user{stored[0], stored[1], stored[2], stored[3], static_cast<std::uint32_t>(buffers), static_cast<std::uint32_t>(buffers >> 32u), static_cast<std::uint32_t>(attributes), static_cast<std::uint32_t>(attributes >> 32u), FetchVertexOffset, 0u};
        const auto pointer = blocks.data.At(0);
        RegisterWrites registers{{0x8b, static_cast<std::uint32_t>(user.size()) << 1u}};
        for (std::uint32_t word = 0; word < user.size(); ++word) registers.emplace_back(0x8cu + word, user[word]);
        for (const auto& entry : RegisterWrites{{0x00b, 8u << 1u}, {0x00c, 0}, {0x00d, 0}, {0x00e, static_cast<std::uint32_t>(pointer)}, {0x00f, static_cast<std::uint32_t>(pointer >> 32u)}}) registers.push_back(entry);
        SubmitDraws(reinterpret_cast<std::uintptr_t>(FetchVertex.data()), reinterpret_cast<std::uintptr_t>(ReadPixel.data()), {{0x242, 4}, {0x24a, FetchIndexOffset}}, registers, blocks.target, draws);
        break;
    }
    default: {
        const auto constants = RawDescriptor(blocks.ring.At(RingSlotsOffset + OuterSlot(state) * RingSlotBytes), ConstantRecords);
        SubmitDraws(reinterpret_cast<std::uintptr_t>(PlainVertex.data()), reinterpret_cast<std::uintptr_t>(RingPixel.data()), {{0x242, 4}}, {{0x8b, 0}, {0x8c, 0}, {0x00b, 4u << 1u}, {0x00c, constants[0]}, {0x00d, constants[1]}, {0x00e, constants[2]}, {0x00f, constants[3]}}, blocks.target, draws);
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
    if (state.kind == Kind::Fetch) {
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

void InjectedCommit(Arms& arms) {
    Require(DrawRecords::VerifyEvery() >= 2, "inject commit: the committer verify needs APS5_VERIFY_DRAW_RECORDS of 2 or more, so hits reach the committer");
    DrawRecords::CorruptCommits(true);
    arms.Run("a corrupted committer answer", Alternating(Kind::Constants, 8), DrawsPerSubmission, [&](const Totals& now, const Totals& base) {
        Require(DrawRecords::Differences(DrawRecords::Field::CommitBindings) != 0 && now.Since(base, Counter::CommitDemoted) != 0, "inject commit: the committer verify did not catch a corrupted buffer binding in the committer answer" + Describe(now, base));
        Require(now.Since(base, Counter::CommitBuilt) != 0, "inject commit: today's path did not rebuild the committer half after a demotion" + Describe(now, base));
    });
    DrawRecords::CorruptCommits(false);
}

}

int main(int argc, char** argv) {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        const bool inject = argc > 1 && std::string_view(argv[1]) == "--inject";
        const bool injectCommit = argc > 1 && std::string_view(argv[1]) == "--inject-commit";
        using AgcDriver::DriverDetail::DrawFinisher;
        using AgcDriver::DriverDetail::DrawPipeline;
        const bool leanEnvironment = RuntimeSwitches::Set(RuntimeSwitch::LeanFront);
        const bool recordsEnvironment = RuntimeSwitches::Set(RuntimeSwitch::DrawRecords);
        Arms arms;
        arms.ring = DrawFinisher::Enabled() && DrawFinisher::Overlap() && DrawPipeline::Depth() != 0;
        Register(PlainVertexHeader, PlainVertex.data(), sizeof(PlainVertex), 2);
        Register(ConstantPixelHeader, ConstantPixel.data(), sizeof(ConstantPixel), 1);
        Register(TableVertexHeader, TableVertex.data(), sizeof(TableVertex), 2);
        Register(ReadPixelHeader, ReadPixel.data(), sizeof(ReadPixel), 1);
        Register(RingPixelHeader, RingPixel.data(), sizeof(RingPixel), 1);
        RegisterFetchVertex();
        Blocks blocks;
        Scene = &blocks;
        Prepare();
        if (inject) {
            Injected(arms);
        } else if (injectCommit) {
            InjectedCommit(arms);
        } else {
            Hits(arms);
            Ring(arms);
            Formats(arms);
            Misses(arms);
            Fallbacks(arms);
        }
        DrawRecords::Report(true);
        RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, leanEnvironment);
        RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, recordsEnvironment);
        AgcDriverShutdown_nid_postfix();
        std::cout << (inject || injectCommit ? "the draw record verify caught a corrupted answer and today's path drew" : "draw record hits render as today's path for moved constants, descriptors and vertex buffers, relocate inline buffer descriptors read through and bound, patch read-only buffer bases and sizes moved inside captured runs, pick the variant of a changed format word, record through the committer half with pushed constants, held targets and per-batch keeps, and fall back on structural changes, writes after the fetch and failed committer checks") << (arms.ring ? "" : " (no finisher ring: records never ran)") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { AgcDriverShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
