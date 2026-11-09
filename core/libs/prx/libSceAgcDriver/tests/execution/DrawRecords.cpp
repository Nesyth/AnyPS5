#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRecords.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/LeanDraw.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/LeanFront.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::DriverDetail::DrawRecord;
using AgcDriver::DriverDetail::DrawRecordKey;
using AgcDriver::DriverDetail::DrawRecordSink;
using AgcDriver::DriverDetail::DrawRecordStore;
using AgcDriver::DriverDetail::LeanShape;
namespace DrawRecords = AgcDriver::DriverDetail::DrawRecords;
namespace LeanFront = AgcDriver::DriverDetail::LeanFront;
using Counter = DrawRecords::Counter;

AgcDriver::Pm4::DrawParameters Parameters(std::uint64_t indexAddress) {
    AgcDriver::Pm4::DrawParameters parameters{};
    parameters.indexAddress = indexAddress;
    parameters.indexCount = 36;
    parameters.indexSize = 2;
    parameters.instanceCount = 1;
    parameters.flags = 0;
    parameters.indexed = true;
    return parameters;
}

std::shared_ptr<const DrawRecord> Record(const DrawRecordKey& key, const std::shared_ptr<const LeanShape>& shape, std::uint32_t content, std::size_t words = 16) {
    DrawRecordSink sink;
    sink.queued = true;
    sink.shape = key.shape;
    sink.contentKey.assign(words, content);
    sink.writes = {{0x1000, 0x2000}};
    return DrawRecord::Build(key, shape, 1, 1, 0, sink);
}

std::vector<DrawRecordKey> KeysInOneSet(std::size_t count) {
    std::vector<DrawRecordKey> keys;
    const auto first = DrawRecordKey::For(0x51, Parameters(0x100000));
    const auto set = DrawRecordStore::SetOf(first.Probe());
    keys.push_back(first);
    for (std::uint64_t address = 0x100100; keys.size() < count; address += 0x100) {
        const auto key = DrawRecordKey::For(0x51, Parameters(address));
        if (DrawRecordStore::SetOf(key.Probe()) == set) keys.push_back(key);
    }
    return keys;
}

void Keys() {
    const auto base = DrawRecordKey::For(0x51, Parameters(0x100000));
    Require(base == DrawRecordKey::For(0x51, Parameters(0x100000)) && base.Probe() == DrawRecordKey::For(0x51, Parameters(0x100000)).Probe(), "equal draws give different record keys");
    std::vector<DrawRecordKey> changed;
    changed.push_back(DrawRecordKey::For(0x52, Parameters(0x100000)));
    changed.push_back(DrawRecordKey::For(0x51, Parameters(0x100040)));
    for (int field = 0; field < 6; ++field) {
        auto parameters = Parameters(0x100000);
        if (field == 0) parameters.indexCount = 30;
        if (field == 1) parameters.indexSize = 4;
        if (field == 2) parameters.instanceCount = 2;
        if (field == 3) parameters.firstVertex = 3;
        if (field == 4) parameters.firstInstance = 1;
        if (field == 5) parameters.indexed = false;
        changed.push_back(DrawRecordKey::For(0x51, parameters));
    }
    for (std::size_t index = 0; index < changed.size(); ++index) {
        Require(!(changed[index] == base) && changed[index].Probe() != base.Probe(), "a record key ignores draw field " + std::to_string(index));
    }
    DrawRecordSink sink;
    sink.contentKey = {1, 2, 3};
    sink.vertexLayout = {4};
    const auto content = sink.Content();
    sink.vertexInputs = {9, 9, 9};
    Require(sink.Content() == content && sink.Inputs() != DrawRecordSink{}.Inputs(), "vertex fetch bases moved a record's content or left its inputs alone");
    sink.contentKey = {1, 2};
    sink.vertexLayout = {3, 4};
    Require(sink.Content() != content, "moving a word between the content key and the vertex layout kept the content hash");
}

void Bounds() {
    DrawRecordStore store;
    const auto evictedBase = DrawRecords::Total(Counter::IdentitiesEvicted);
    const auto keys = KeysInOneSet(DrawRecordStore::Ways + 1);
    auto shape = std::make_shared<LeanShape>();
    for (std::size_t index = 0; index < DrawRecordStore::Ways; ++index) {
        store.Place(keys[index], keys[index].Probe(), 10, 1);
        Require(store.Install(Record(keys[index], shape, static_cast<std::uint32_t>(index)), 10, 1) == DrawRecordStore::Installed::Yes, "a record for a placed identity was not installed");
    }
    Require(store.Identities() == DrawRecordStore::Ways && store.Records() == DrawRecordStore::Ways, "a full set does not hold one record per way");
    Require(store.Find(keys[0], keys[0].Probe(), 10) != nullptr, "the first identity of a full set was not found");
    store.Place(keys.back(), keys.back().Probe(), 10, 1);
    Require(store.Identities() == DrawRecordStore::Ways, "placing into a full set grew the store past its ways");
    Require(store.Find(keys[1], keys[1].Probe(), 10) == nullptr, "the least recently seen identity of a full set was kept over a new one");
    Require(store.Find(keys[0], keys[0].Probe(), 10) != nullptr && store.Find(keys.back(), keys.back().Probe(), 10) != nullptr, "a recently seen or new identity was evicted");
    Require(DrawRecords::Total(Counter::IdentitiesEvicted) == evictedBase + 1, "the eviction of an identity with a record was not counted");
    Require(store.Install(Record(keys[1], shape, 1), 10, 1) == DrawRecordStore::Installed::NoEntry, "a record for an evicted identity was installed");
    auto& entry = *store.Find(keys[0], keys[0].Probe(), 10);
    for (std::uint32_t content = 10; content < 10 + DrawRecordStore::Variants + 2; ++content) {
        Require(store.Install(Record(keys[0], shape, content), 10, 1) == DrawRecordStore::Installed::Yes, "a variant was not installed");
    }
    Require(entry.records->records.size() == DrawRecordStore::Variants, "an identity holds " + std::to_string(entry.records->records.size()) + " variants");
    Require(entry.records->records.front()->contentKey.front() == 10 + DrawRecordStore::Variants + 1, "the newest variant is not first");
    const auto held = store.Records();
    Require(store.Install(Record(keys[0], shape, 10 + DrawRecordStore::Variants + 1), 10, 1) == DrawRecordStore::Installed::Yes && store.Records() == held, "a record with known content added a variant instead of replacing it");
    auto other = std::make_shared<LeanShape>();
    Require(store.Install(Record(keys[0], other, 77), 10, 1) == DrawRecordStore::Installed::Yes && entry.records->records.size() == 1, "a record of a rebuilt shape kept the variants of the old shape");
    std::size_t bytes = 0;
    for (const auto& record : entry.records->records) bytes += record->Bytes();
    Require(store.Bytes() >= bytes && store.Bytes() > 0, "the store does not count its record bytes");
}

void Eviction() {
    DrawRecordStore store;
    const auto keys = KeysInOneSet(DrawRecordStore::Ways + 2);
    auto shape = std::make_shared<LeanShape>();
    for (std::size_t index = 0; index < DrawRecordStore::Ways; ++index) {
        store.Place(keys[index], keys[index].Probe(), index == 2 ? 1 : 10, 1);
        if (index != 3) store.Install(Record(keys[index], shape, static_cast<std::uint32_t>(index)), 10, 1);
    }
    for (std::size_t index = 0; index < DrawRecordStore::Ways; ++index) {
        if (index != 2) store.Find(keys[index], keys[index].Probe(), 10);
    }
    store.Place(keys[4], keys[4].Probe(), 10, 1);
    Require(store.Find(keys[2], keys[2].Probe(), 10) == nullptr, "an identity not seen for more than " + std::to_string(DrawRecordStore::KeepPresents) + " presents was kept over recent ones");
    store.Place(keys[5], keys[5].Probe(), 10, 1);
    Require(store.Find(keys[3], keys[3].Probe(), 10) == nullptr || store.Find(keys[4], keys[4].Probe(), 10) == nullptr, "a new identity evicted a recent identity with a record while one without a record was in the set");
    Require(store.Find(keys[0], keys[0].Probe(), 10) != nullptr && store.Find(keys[1], keys[1].Probe(), 10) != nullptr, "identities with records were evicted before identities without one");
    DrawRecordStore capped;
    capped.SetByteCap(Record(keys[0], shape, 1, 64)->Bytes() + Record(keys[1], shape, 2, 64)->Bytes() / 2);
    capped.Place(keys[0], keys[0].Probe(), 1, 1);
    capped.Place(keys[1], keys[1].Probe(), 1, 1);
    Require(capped.Install(Record(keys[0], shape, 1, 64), 1, 1) == DrawRecordStore::Installed::Yes, "a record under the byte cap was refused");
    Require(capped.Install(Record(keys[1], shape, 2, 64), 1, 1) == DrawRecordStore::Installed::OverCap, "a record over the byte cap was installed");
    capped.Find(keys[1], keys[1].Probe(), 10);
    Require(capped.Install(Record(keys[1], shape, 2, 64), 10, 1) == DrawRecordStore::Installed::Yes, "a record that fits once aged identities go was refused");
    Require(capped.Find(keys[0], keys[0].Probe(), 10) == nullptr && capped.Bytes() <= capped.ByteCap(), "the byte cap kept an aged identity or was exceeded");
}

void Invalidation() {
    const auto keys = KeysInOneSet(3);
    auto shape = std::make_shared<LeanShape>();
    DrawRecordStore store;
    for (const auto& key : keys) store.Place(key, key.Probe(), 1, 1);
    store.Install(Record(keys[0], shape, 1), 1, 1);
    store.Install(Record(keys[1], shape, 2), 1, 1);
    auto& carried = *store.Find(keys[0], keys[0].Probe(), 1);
    carried.lastSeq = 20;
    const std::weak_ptr<const DrawRecord> inFlight = carried.records->records.front();
    const std::weak_ptr<const DrawRecord> idle = store.Find(keys[1], keys[1].Probe(), 1)->records->records.front();
    const auto clearedBase = DrawRecords::Total(Counter::RecordsCleared);
    store.Clear(10);
    Require(store.Identities() == 0 && store.Records() == 0 && store.Bytes() == 0 && store.Find(keys[0], keys[0].Probe(), 1) == nullptr, "a cleared store still holds identities");
    Require(DrawRecords::Total(Counter::RecordsCleared) == clearedBase + 2, "the records a clear dropped were not counted");
    Require(!inFlight.expired() && idle.expired(), "a clear freed a record a job ahead of the finisher front carries, or kept one no job carries");
    store.Release(20);
    Require(!inFlight.expired(), "a record was freed while the finisher front had not passed the last job that carried it");
    store.Release(21);
    Require(inFlight.expired() && store.RetiredSets() == 0, "a record was kept after the finisher front passed every job that carried it");
    auto& demoted = store.Place(keys[2], keys[2].Probe(), 1, 1);
    store.Install(Record(keys[2], shape, 3), 1, 1);
    Require(store.Demote(keys[2], keys[2].Probe(), 1) && demoted.refused && demoted.records == nullptr, "a demoted identity kept its records");
    Require(store.Install(Record(keys[2], shape, 3), 1, 1) == DrawRecordStore::Installed::Refused, "a record was installed for a refused identity");
    store.Clear(1);
    auto& renewed = store.Place(keys[2], keys[2].Probe(), 1, 1);
    Require(!renewed.refused && store.Install(Record(keys[2], shape, 3), 1, 1) == DrawRecordStore::Installed::Yes, "a refused identity stayed refused after the store was cleared");
    renewed.buildSeq = 5;
    Require(renewed.BuildPending(5) && !renewed.BuildPending(6), "a build is pending past the finisher front, or not pending before it");
    const std::weak_ptr<const DrawRecord> held = renewed.records->records.front();
    renewed.lastSeq = 100;
    store.Reset();
    Require(held.expired() && store.Identities() == 0 && store.RetiredSets() == 0, "a reset kept records");
}

alignas(256) constexpr std::array<std::uint32_t, 16> VertexCode{
    0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
};
alignas(256) constexpr std::array<std::uint32_t, 10> PixelCode{0xf4000100u, 0xfa000000u, 0xbf8cc07fu, 0x7e000204u, 0x06000002u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u};
alignas(256) std::array<std::array<float, 64>, 2> PixelConstants{};
alignas(256) std::array<std::byte, 1024> ColorMemory{};
constexpr std::uint32_t Submissions = 6;
constexpr std::uint32_t DrawsPerSubmission = 8;

struct Header {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(8) Header VertexHeader{};
alignas(8) Header PixelHeader{};

void Registered(Header& header, const std::uint32_t* code, std::size_t bytes, std::uint8_t type) {
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(bytes);
    header.shader.code = code;
    header.shader.type = type;
    header.shader.user_data = &header.userData;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

void Append(std::vector<std::uint32_t>& commands, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    commands.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    commands.insert(commands.end(), payload);
}

std::vector<std::uint32_t> Submission(std::uint32_t table, std::uint32_t vertices, std::uint32_t draws) {
    const auto color = reinterpret_cast<std::uintptr_t>(ColorMemory.data());
    const auto vertex = reinterpret_cast<std::uintptr_t>(VertexCode.data());
    const auto pixel = reinterpret_cast<std::uintptr_t>(PixelCode.data());
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
    std::vector<std::uint32_t> commands;
    Append(commands, 0x79, {0x242, 4});
    for (const auto& [offset, value] : context) Append(commands, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) Append(commands, 0x76, {offset, value});
    const auto address = reinterpret_cast<std::uintptr_t>(PixelConstants[table].data());
    Append(commands, 0x76, {0x00c, static_cast<std::uint32_t>(address)});
    Append(commands, 0x76, {0x00d, static_cast<std::uint32_t>(address >> 32u)});
    Append(commands, 0x76, {0x00e, std::bit_cast<std::uint32_t>(0.25f)});
    for (std::uint32_t draw = 0; draw < draws; ++draw) Append(commands, 0x2d, {vertices, 2});
    return commands;
}

void Submit(std::uint32_t table, std::uint32_t vertices, std::uint32_t draws, const std::string& arm) {
    static std::vector<std::uint32_t> commands;
    commands = Submission(table, vertices, draws);
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "a draw submission failed" + arm);
    AgcDriverWaitIdle_nid_postfix();
}

struct Totals {
    std::array<std::uint64_t, static_cast<std::size_t>(Counter::Count)> counters{};
    std::uint64_t differences = 0;
    std::uint64_t fetches = 0;
    std::uint64_t lookups = 0;

    static Totals Now() {
        Totals totals;
        for (std::size_t index = 0; index < totals.counters.size(); ++index) totals.counters[index] = DrawRecords::Total(static_cast<Counter>(index));
        totals.differences = DrawRecords::Differences();
        totals.fetches = DrawRecords::Checked(DrawRecords::Item::Fetch);
        totals.lookups = DrawRecords::Checked(DrawRecords::Item::Lookup);
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

int Device() {
    bool libraries = false;
    {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        libraries = device->GraphicsPipelineLibraries();
    }
    if (!libraries && std::getenv("APS5_NO_GPL") == nullptr) return VulkanTestSkipped;
    using AgcDriver::DriverDetail::DrawFinisher;
    using AgcDriver::DriverDetail::DrawPipeline;
    using AgcDriver::Graphics::RuntimeSwitch;
    namespace RuntimeSwitches = AgcDriver::Graphics::RuntimeSwitches;
    const bool ring = DrawFinisher::Enabled() && DrawFinisher::Overlap() && DrawPipeline::Depth() != 0;
    const bool leanEnvironment = RuntimeSwitches::Set(RuntimeSwitch::LeanFront);
    const bool recordsEnvironment = RuntimeSwitches::Set(RuntimeSwitch::DrawRecords);
    Registered(VertexHeader, VertexCode.data(), sizeof(VertexCode), 2);
    Registered(PixelHeader, PixelCode.data(), sizeof(PixelCode), 1);
    for (auto& table : PixelConstants) table[0] = 0.25f;
    std::array<std::byte, 1024> todayColors{};
    for (const int arm : {0, 1, 2}) {
        const bool lean = arm != 0;
        const bool records = arm == 2;
        RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, lean);
        RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, records);
        const std::string name = arm == 0 ? " (today's front)" : arm == 1 ? " (lean front)" : " (lean front with draw records)";
        ColorMemory.fill(std::byte{0});
        const auto base = Totals::Now();
        for (std::uint32_t submission = 0; submission < Submissions; ++submission) Submit(submission & 1u, 3, DrawsPerSubmission, name);
        AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(ColorMemory.data()), ColorMemory.size());
        unsigned brightest = 0;
        for (const auto value : ColorMemory) brightest = std::max(brightest, static_cast<unsigned>(std::to_integer<std::uint8_t>(value)));
        Require(brightest + 2 >= 128 && brightest <= 130, "the draws wrote " + std::to_string(brightest) + ", expected 128" + name);
        if (arm == 0) todayColors = ColorMemory;
        else Require(ColorMemory == todayColors, "the draws rendered differently from today's front" + name);
        const auto now = Totals::Now();
        Require(now.differences == base.differences, "the draw record verify found differences" + name);
        const auto draws = now.Since(base, Counter::Draws);
        const auto off = now.Since(base, Counter::OffTodaysDraw) + now.Since(base, Counter::OffIndirect) + now.Since(base, Counter::OffLeanVerify) + now.Since(base, Counter::OffInline);
        if (!records) {
            Require(draws == 0, "draws were counted for draw records while records are off" + name + Describe(now, base));
            continue;
        }
        if (!ring) {
            Require(now.Since(base, Counter::Hits) == 0 && off == draws, "draws without the finisher ring were not counted as draws without records" + name + Describe(now, base));
            continue;
        }
        Require(draws == Submissions * DrawsPerSubmission, "draw records counted " + std::to_string(draws) + " draws, expected " + std::to_string(Submissions * DrawsPerSubmission) + name + Describe(now, base));
        std::uint64_t outcomes = 0;
        for (auto counter = static_cast<std::size_t>(Counter::Hits); counter <= static_cast<std::size_t>(Counter::OffInline); ++counter) {
            if (static_cast<Counter>(counter) == Counter::InputRefetches || static_cast<Counter>(counter) == Counter::HitsOnTodaysPath) continue;
            outcomes += now.Since(base, static_cast<Counter>(counter));
        }
        Require(outcomes == draws, "the hits, misses and draws without records add up to " + std::to_string(outcomes) + " of " + std::to_string(draws) + " draws" + name + Describe(now, base));
        const auto jobs = draws - off;
        const auto hits = now.Since(base, Counter::Hits);
        Require(jobs > DrawsPerSubmission && hits + DrawsPerSubmission >= jobs, "draw records hit " + std::to_string(hits) + " of " + std::to_string(jobs) + " finisher jobs of one identity whose user data pointer moves between submissions" + name + Describe(now, base));
        const auto verified = now.Since(base, Counter::HitsOnTodaysPath);
        const bool verifiedAsDue = DrawRecords::VerifyEvery() == 0 ? verified == 0 : DrawRecords::VerifyEvery() == 1 ? verified == hits : verified != 0 && verified < hits;
        Require(now.Since(base, Counter::MissNewIdentity) == 1 && now.Since(base, Counter::RecordsInstalled) >= 1 && now.Since(base, Counter::PlansBuilt) >= 1 && verifiedAsDue, "one identity did not install a record with a fetch plan, or its hits did not run the record path except where the verify ran today's" + name + Describe(now, base));
        Require(now.Since(base, Counter::MissContentDiffers) == 0 && now.Since(base, Counter::InputRefetches) == 0, "draws of one identity with moved constants missed on content or moved vertex inputs" + name + Describe(now, base));
        Require(DrawRecords::VerifyEvery() == 0 || (now.fetches > base.fetches && now.lookups > base.lookups), "the draw record verify checked no hit" + name);
        const auto other = Totals::Now();
        Submit(0, 6, 2, name);
        Submit(0, 6, 2, name);
        Submit(0, 6, 2, name);
        const auto counted = Totals::Now();
        Require(counted.Since(other, Counter::MissNewIdentity) == 1 && counted.Since(other, Counter::Hits) >= 1, "a draw with another vertex count did not start one new identity that later hits" + name + Describe(counted, other));
        RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, false);
        Submit(0, 3, 2, name);
        const auto off0 = Totals::Now();
        Require(off0.Since(counted, Counter::Draws) == 0 && off0.Since(counted, Counter::RecordModeChanges) == 1 && off0.Since(counted, Counter::RecordsCleared) >= 2, "turning draw records off did not clear the records and stop counting draws" + name + Describe(off0, counted));
        RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, true);
        Submit(0, 3, 2, name);
        const auto on = Totals::Now();
        Require(on.Since(off0, Counter::RecordModeChanges) == 1 && on.Since(off0, Counter::MissNewIdentity) == 1 && on.Since(off0, Counter::Hits) == 0, "turning draw records on again did not start from an empty store" + name + Describe(on, off0));
        Require(on.differences == base.differences, "the draw record verify found differences after the switch flips" + name);
        DrawRecords::Report(true);
    }
    RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, leanEnvironment);
    RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, recordsEnvironment);
    AgcDriverShutdown_nid_postfix();
    return 0;
}

}

int main() {
    try {
        Keys();
        Bounds();
        Eviction();
        Invalidation();
        const auto device = Device();
        std::cout << "draw record keys, store bounds, eviction and invalidation passed" << (device == 0 ? "; repeated lean draws hit their identity records" : "; the device part was skipped") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
