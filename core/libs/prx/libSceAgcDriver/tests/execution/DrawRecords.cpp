#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRecords.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/LeanDraw.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/StagePlans.hpp"
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
using AgcDriver::DriverDetail::DispatchVariant;
using AgcDriver::DriverDetail::DrawRecord;
using AgcDriver::DriverDetail::DrawRecordKey;
using AgcDriver::DriverDetail::DrawRecordPlan;
using AgcDriver::DriverDetail::DrawRecordProbe;
using AgcDriver::DriverDetail::DrawRecordSet;
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

alignas(64) std::array<std::uint32_t, 16> ProbedMemory{};

std::uint64_t ProbedAt(std::size_t word) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(ProbedMemory.data() + word));
}

std::uint32_t ProbedBase(std::size_t word) {
    return static_cast<std::uint32_t>(ProbedAt(word));
}

std::shared_ptr<const DrawRecordPlan> ProbedPlan(std::array<std::uint32_t, 6> fixed, std::array<std::uint32_t, 2> based, std::uint32_t data, std::uint32_t base = ProbedBase(4)) {
    auto plan = std::make_shared<DrawRecordPlan>();
    plan->stages.emplace_back();
    auto& stage = plan->stages.emplace_back();
    stage.userData = {base};
    auto variant = std::make_shared<DispatchVariant>();
    const auto shift = static_cast<std::uint64_t>(base & ~3u) - (ProbedBase(4) & ~3u);
    variant->runs = {{ProbedAt(0), ProbedAt(4)}, {ProbedAt(4) + shift, ProbedAt(6) + shift}, {ProbedAt(8), ProbedAt(9)}, {ProbedAt(10), ProbedAt(12)}};
    variant->words = {fixed[0], fixed[1], fixed[2], fixed[3], based[0], based[1], data, fixed[4], fixed[5]};
    variant->dataPositions = {6};
    stage.runBases = {-1, 0, -1, -1};
    stage.variant = std::move(variant);
    return plan;
}

std::shared_ptr<const DrawRecord> ProbedRecord(std::shared_ptr<const DrawRecordPlan> plan, const DrawRecordSet* kept) {
    auto record = std::make_shared<DrawRecord>();
    if (kept != nullptr) record->probes = AgcDriver::DriverDetail::DrawRecordProbes(*plan, kept);
    record->plan = std::move(plan);
    return record;
}

const DrawRecord* Picked(const DrawRecordSet& set, std::uint32_t base) {
    const std::array<std::vector<std::uint32_t>, 2> users{std::vector<std::uint32_t>{}, std::vector<std::uint32_t>{base}};
    for (const auto& record : set.records) {
        if (AgcDriver::DriverDetail::DrawRecordProbesHold(record->probes, users)) return record.get();
    }
    return nullptr;
}

void Probes() {
    using AgcDriver::DriverDetail::DrawRecordProbes;
    const auto first = ProbedPlan({1, 2, 3, 4, 11, 12}, {5, 6}, 7);
    const auto second = ProbedPlan({1, 2, 9, 4, 11, 12}, {5, 6}, 70, ProbedBase(6));
    DrawRecordSet set;
    set.records.push_back(ProbedRecord(first, nullptr));
    Require(set.records.front()->probes.empty(), "a record built without variants beside it got probe words");
    const auto probes = DrawRecordProbes(*second, &set);
    Require(probes.size() == 1 && probes.front().address == ProbedAt(2) && probes.front().value == 9 && probes.front().base < 0 && probes.front().stage == 1, "the probes of a variant are not exactly the fixed structural word it differs in (equal words of a run that moved with its user word and data words must not be probed): " + std::to_string(probes.size()) + " probes");
    set.records.insert(set.records.begin(), ProbedRecord(second, &set));
    ProbedMemory = {1, 2, 3, 4, 5, 6, 0, 0, 7, 0, 11, 12, 0, 0, 0, 0};
    Require(Picked(set, ProbedBase(4)) == set.records.back().get(), "the pre-check did not pass over the newer variant to the record of the memory's structure");
    ProbedMemory[2] = 9;
    ProbedMemory[8] = 78;
    Require(Picked(set, ProbedBase(4)) == set.records.front().get(), "the pre-check did not pick the newest variant in its own memory, whatever its data words hold");
    const auto based = ProbedPlan({1, 2, 9, 4, 11, 12}, {5, 66}, 7, ProbedBase(8));
    const auto probed = DrawRecordProbes(*based, &set);
    const auto relative = std::find_if(probed.begin(), probed.end(), [](const DrawRecordProbe& probe) { return probe.base == 0; });
    Require(relative != probed.end() && relative->value == 66 && relative->stage == 1 && relative->address + (ProbedBase(4) & ~3u) == ProbedAt(5), "a variant differing in a word of a run based on a user word was not probed relative to that user word");
    const std::vector<DrawRecordProbe> follow{*relative};
    const std::array<std::vector<std::uint32_t>, 2> moved{std::vector<std::uint32_t>{}, std::vector<std::uint32_t>{ProbedBase(10)}};
    ProbedMemory[11] = 66;
    Require(AgcDriver::DriverDetail::DrawRecordProbesHold(follow, moved), "a probe relative to a user word did not follow the draw's moved user word");
    ProbedMemory[11] = 12;
    Require(!AgcDriver::DriverDetail::DrawRecordProbesHold(follow, moved), "a probe relative to a user word held with another value at the moved address");
    const auto third = ProbedPlan({21, 22, 23, 24, 25, 26}, {5, 6}, 7);
    const auto many = DrawRecordProbes(*third, &set);
    Require(many.size() == DrawRecordStore::ProbesPerVariant, "a variant differing from two variants in every fixed word got " + std::to_string(many.size()) + " probes, expected the per-variant limit against the newer one, whose probes tell the older one apart too");
    ProbedMemory = {21, 22, 23, 24, 0, 0, 0, 0, 0, 0, 25, 26, 0, 0, 0, 0};
    Require(AgcDriver::DriverDetail::DrawRecordProbesHold(many, moved), "the probes of a variant failed in its own memory");
    ProbedMemory = {1, 2, 3, 4, 5, 6, 0, 0, 7, 0, 11, 12, 0, 0, 0, 0};
    Require(!AgcDriver::DriverDetail::DrawRecordProbesHold(many, moved), "the probes taken against the newer variant held in the memory of the older one they were meant to tell apart");
    DrawRecordSet pair;
    pair.records.push_back(ProbedRecord(ProbedPlan({1, 2, 9, 4, 11, 12}, {5, 6}, 7), nullptr));
    pair.records.push_back(ProbedRecord(ProbedPlan({5, 6, 9, 4, 11, 12}, {5, 6}, 7), nullptr));
    const auto apart = DrawRecordProbes(*ProbedPlan({5, 6, 9, 4, 11, 40}, {5, 6}, 7), &pair);
    const auto late = std::find_if(apart.begin(), apart.end(), [](const DrawRecordProbe& probe) { return probe.base < 0 && probe.address == ProbedAt(11) && probe.value == 40; });
    Require(apart.size() == DrawRecordStore::ProbesPerVariant + 1 && late != apart.end(), "an older variant that the probes against a newer one do not tell apart got no probe of its own: " + std::to_string(apart.size()) + " probes");
    ProbedMemory = {5, 6, 9, 4, 5, 6, 0, 0, 7, 0, 11, 12, 0, 0, 0, 0};
    Require(!AgcDriver::DriverDetail::DrawRecordProbesHold(apart, moved), "the probes of a variant held in the memory of an older variant they were meant to tell apart");
    ProbedMemory[11] = 40;
    Require(AgcDriver::DriverDetail::DrawRecordProbesHold(apart, moved), "the probes of a variant failed in its own memory");
    Require(!AgcDriver::DriverDetail::DrawRecordProbesHold(std::vector<DrawRecordProbe>{{0, 0, 0, -1}}, moved), "a probe at an unmapped address held");
}

void Twins() {
    using AgcDriver::DriverDetail::DrawRecordTwin;
    const auto key = DrawRecordKey::For(0x51, Parameters(0x100000));
    auto shape = std::make_shared<LeanShape>();
    DrawRecordSet set;
    for (std::uint32_t content = 1; content <= 3; ++content) set.records.push_back(Record(key, shape, content));
    const auto& last = *set.records.back();
    const auto signature = last.signature;
    const auto other = signature + 1;
    const auto twin = DrawRecordTwin(set, last.content, last.generation, last.deviceSerial, &signature);
    Require(twin.has_value() && *twin == 2, "the twin of the last variant was not found at its place in the set");
    Require(!DrawRecordTwin(set, last.content, last.generation, last.deviceSerial, &other).has_value(), "a record of another plan structure was taken for a twin");
    Require(DrawRecordTwin(set, last.content, last.generation, last.deviceSerial, nullptr) == twin, "a draw without a fetch plan did not find the record of its content");
    Require(!DrawRecordTwin(set, last.content, last.generation + 1, last.deviceSerial, &signature).has_value() && !DrawRecordTwin(set, last.content, last.generation, last.deviceSerial + 1, &signature).has_value(), "a record of another switch generation or device was taken for a twin");
}

alignas(64) std::array<std::uint32_t, 64> PlanMemory{};
constexpr std::size_t PlanTable = 0;
constexpr std::size_t PlanMovedTable = 16;
constexpr std::size_t PlanFixedRun = 40;
constexpr std::uint32_t PlanFormat = 0x31016facu;

std::uint64_t PlanAt(std::size_t word) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(PlanMemory.data() + word));
}

struct PlanDraw {
    std::size_t table = PlanTable;
    std::uint32_t buffer = 0x1000;
    std::uint32_t records = 16;
    std::uint32_t format = PlanFormat;
    std::uint32_t data = 7;
    std::uint32_t forwarded = 3;
};

std::vector<std::uint32_t> PlanUsers(const PlanDraw& draw) {
    const auto table = PlanAt(draw.table);
    return {static_cast<std::uint32_t>(table), static_cast<std::uint32_t>(table >> 32u), draw.forwarded, 0xabcdu};
}

void WritePlanMemory(const PlanDraw& draw) {
    const std::array<std::uint32_t, 8> table{draw.buffer, 0x00100000u, draw.records, draw.format, 0x11u, 0x22u, 0x33u, 0x44u};
    std::copy(table.begin(), table.end(), PlanMemory.begin() + static_cast<std::ptrdiff_t>(draw.table));
    PlanMemory[PlanFixedRun] = draw.data;
    PlanMemory[PlanFixedRun + 1] = 0x5150u;
}

std::vector<std::uint32_t> PlanWords(const PlanDraw& draw) {
    return {draw.buffer, 0x00100000u, draw.records, draw.format, 0x11u, 0x22u, 0x33u, 0x44u, draw.data, 0x5150u};
}

AgcDriver::DriverDetail::DrawRecordStage PlanStage(const PlanDraw& draw, std::uint64_t variantId) {
    using ShaderRecompiler::DescriptorRole;
    auto compiled = std::make_shared<ShaderRecompiler::RecompileResult>();
    compiled->variantId = variantId;
    compiled->bindings.resize(2);
    compiled->bindings[0].role = DescriptorRole::GuestBuffers;
    compiled->bindings[0].count = 1;
    compiled->bindings[0].guestDescriptor = {draw.buffer, 0x00100000u, draw.records, draw.format};
    compiled->bindings[0].bufferWritten = {false};
    compiled->bindings[1].role = DescriptorRole::FlattenedSrt;
    compiled->bindings[1].count = 1;
    compiled->bindings[1].guestDescriptor = {draw.data};
    AgcDriver::DriverDetail::DrawRecordStage stage;
    stage.stage = ShaderRecompiler::ShaderStage::Fragment;
    stage.compiled = compiled;
    stage.userData = PlanUsers(draw);
    stage.relocatableWords = 1u;
    stage.walkedWords = 3u;
    stage.valueUseWords = 2u;
    stage.forwardedWords = 4u;
    stage.userDataMask = 7u;
    stage.runBases = {0, -1};
    stage.targets = {{0, 0, 0}, {2, 0, 2}, {8, 1, 0}};
    auto variant = std::make_shared<DispatchVariant>();
    variant->runs = {{PlanAt(draw.table), PlanAt(draw.table + 8)}, {PlanAt(PlanFixedRun), PlanAt(PlanFixedRun + 2)}};
    variant->words = PlanWords(draw);
    variant->dataPositions = {0, 2, 8};
    variant->compiled = compiled;
    stage.variant = std::move(variant);
    return stage;
}

std::shared_ptr<const AgcDriver::DriverDetail::StagePlanAnswer> PlanAnswer(const std::shared_ptr<const LeanShape>& shape, std::initializer_list<std::shared_ptr<const AgcDriver::DriverDetail::StagePlan>> stages, std::uint32_t content) {
    auto answer = std::make_shared<AgcDriver::DriverDetail::StagePlanAnswer>();
    answer->shape = shape;
    answer->generation = 1;
    answer->deviceSerial = 1;
    answer->stages.assign(stages.begin(), stages.end());
    DrawRecordKey key{};
    key.shape = 0x77;
    answer->record = Record(key, shape, content);
    return answer;
}

void StagePlanKeys() {
    namespace Plans = AgcDriver::DriverDetail;
    using Plans::StagePlans::Counter;
    using AgcDriver::DriverDetail::StagePlanKey;
    const PlanDraw first{};
    const auto built = Plans::MakeStagePlan(PlanStage(first, 0x51));
    Require(built != nullptr, "a stage plan was not made from a fetch plan stage");
    const auto& layout = *built->layout;
    Require(layout.runs.size() == 2 && layout.runs[0].base == 0 && layout.runs[1].base < 0 && layout.runs[1].offset == PlanAt(PlanFixedRun) && layout.words == 10, "a stage layout does not keep its runs relative to their user word bases");
    Require(layout.runs[0].offset + (PlanUsers(first)[0] & ~3u) == PlanAt(first.table), "a based run's offset does not lead back to its address from the user word");
    Require(layout.keyPositions == std::vector<std::uint32_t>{1, 3, 4, 5, 6, 7, 9} && layout.bitPositions == std::vector<std::uint32_t>{0, 2}, "the key does not hold every run word but the buffer base, its record count and the flattened data word, or the low bits of those two buffer words");
    Require(layout.exactUsers == 2u && layout.bitUsers == 1u, "the key's user words are not the walked high word exactly and the relocatable base by its low bits (the forwarded and the unread words stay out)");
    WritePlanMemory(first);
    std::vector<std::uint32_t> words;
    StagePlanKey key;
    Require(Plans::ReadStageRuns(layout, PlanUsers(first), words) && words == PlanWords(first), "reading a layout's runs at the build's user words did not give the build's words");
    Plans::StageKeyOf(layout, PlanUsers(first), words, key);
    Require(key == built->key, "the key read from memory at the build's user words differs from the plan's");
    PlanDraw moved{PlanMovedTable, 0x2340, 32, PlanFormat, 99, 8};
    WritePlanMemory(moved);
    Require(Plans::ReadStageRuns(layout, PlanUsers(moved), words) && words == PlanWords(moved), "reading a layout's runs did not follow the moved table pointer");
    Plans::StageKeyOf(layout, PlanUsers(moved), words, key);
    Require(key == built->key, "a draw whose table moved, whose constant buffer moved inside it and whose data and forwarded words changed got another key");
    Plans::StagePlanStage stage;
    stage.layouts.push_back(built->layout);
    stage.plans.push_back(built);
    Require(Plans::FindStagePlan(stage, built->layout.get(), key) == built.get(), "the stage plan of a moved draw's key was not found");
    const auto missOf = [&](const PlanDraw& draw, std::vector<std::uint32_t> users) {
        Plans::StageKeyOf(layout, users, PlanWords(draw), key);
        Require(Plans::FindStagePlan(stage, built->layout.get(), key) == nullptr, "a key that differs from the plan's found the plan");
        return Plans::StageKeyMiss(stage, built->layout.get(), key);
    };
    auto format = moved;
    format.format = 0x31014facu;
    Require(missOf(format, PlanUsers(format)) == Counter::MissNewRunWords, "a changed buffer format word was not a new run word key");
    auto high = PlanUsers(moved);
    high[1] ^= 0x10u;
    Require(missOf(moved, high) == Counter::MissNewUserWords, "a changed walked user word was not a new user word key");
    auto aligned = PlanUsers(moved);
    aligned[0] |= 1u;
    Require(missOf(moved, aligned) == Counter::MissNewBits, "a run base with other low bits was not a new low bits key");
    auto tail = moved;
    tail.records = 33;
    Require(missOf(tail, PlanUsers(tail)) == Counter::MissNewBits, "a buffer record count with another tail was not a new low bits key");
    auto offset = moved;
    offset.buffer = 0x2342;
    Require(missOf(offset, PlanUsers(offset)) == Counter::MissNewBits, "a buffer base with other low bits was not a new low bits key");
    auto unread = PlanUsers(moved);
    unread[3] = 0x9999u;
    Plans::StageKeyOf(layout, unread, PlanWords(moved), key);
    Require(Plans::FindStagePlan(stage, built->layout.get(), key) == built.get(), "a user word the stage never reads moved the key");
    auto valued = PlanStage(first, 0x51);
    valued.userData.push_back(0x5000u);
    valued.valueWords = 1u << 4u;
    valued.vertexWords = 1u << 3u;
    valued.userDataMask |= (1u << 3u) | (1u << 4u);
    const auto withValue = Plans::MakeStagePlan(valued);
    Require(withValue != nullptr && withValue->layout->exactUsers == 2u && withValue->layout->bitUsers == ((1u << 4u) | 1u) && withValue->layout->users == 5u, "a user word patched into a bound buffer did not join the key by its low bits, or a vertex table word joined it");
    Plans::StagePlanStage valueStage;
    valueStage.layouts.push_back(withValue->layout);
    valueStage.plans.push_back(withValue);
    const auto valueKey = [&](std::uint32_t value, std::uint32_t vertex) {
        auto users = valued.userData;
        users[3] = vertex;
        users[4] = value;
        Plans::StageKeyOf(*withValue->layout, users, PlanWords(first), key);
        return Plans::FindStagePlan(valueStage, withValue->layout.get(), key) == withValue.get();
    };
    Require(valueKey(0x7340u, 0x1234u), "a moved buffer base in a user word or a moved vertex table pointer gave another key");
    Require(!valueKey(0x7342u, 0xabcdu) && !valueKey(0u, 0xabcdu), "a user word buffer base with other low bits or none at all kept the key");
    auto hidden = PlanStage(first, 0x51);
    hidden.hiddenUserWords = 1;
    const auto withHidden = Plans::MakeStagePlan(hidden);
    Require(withHidden != nullptr && withHidden->layout->exactUsers == 3u && withHidden->layout->bitUsers == 0u, "a hidden user word did not join the key as it is");
}

void StagePlanBuilds() {
    namespace Plans = AgcDriver::DriverDetail;
    std::uint64_t pending = 0;
    Require(Plans::StagePlanBuildDue(pending, 10, 5) && pending == 10, "the first build of a shape was not due");
    Require(!Plans::StagePlanBuildDue(pending, 11, 5) && !Plans::StagePlanBuildDue(pending, 12, 10) && pending == 10, "a second build was due while the first one's job had not finished");
    Require(Plans::StagePlanBuildDue(pending, 13, 11) && pending == 13, "a build was not due once the finisher front passed the last build's job");
}

void StagePlanMerges() {
    namespace Plans = AgcDriver::DriverDetail;
    auto shape = std::make_shared<LeanShape>();
    const auto a = Plans::MakeStagePlan(PlanStage(PlanDraw{}, 0x51));
    Plans::StagePlanMerge merge;
    auto set = Plans::MergeStagePlans(nullptr, *PlanAnswer(shape, {a}, 1), merge);
    Require(set != nullptr && set->stages.size() == 1 && set->stages[0].plans.size() == 1 && set->stages[0].layouts.size() == 1 && set->draws.size() == 1 && merge.added == 1 && merge.layouts == 1 && merge.drawsAdded == 1, "the first build did not make one layout, one stage plan and one draw plan");
    const auto* first = set->stages[0].plans.front().get();
    Require(set->draws.front().record->plan != nullptr && set->draws.front().record->plan->stages.size() == 1 && set->draws.front().record->plan->stages[0].compiled == a->stage.compiled && set->draws.front().record->commit != nullptr, "a draw plan's record does not carry its stage plans' fetch plans and a committer half");
    const auto moved = Plans::MakeStagePlan(PlanStage(PlanDraw{PlanMovedTable, 0x2340, 32, PlanFormat, 99, 8}, 0x51));
    merge = {};
    set = Plans::MergeStagePlans(set.get(), *PlanAnswer(shape, {moved}, 1), merge);
    Require(set->stages[0].plans.size() == 1 && set->draws.size() == 1 && merge.shared == 1 && merge.drawsKnown == 1 && set->stages[0].plans.front().get() == first, "a build of a moved draw with the same key did not share the stage plan and its draw plan");
    PlanDraw other{};
    other.format = 0x31014facu;
    const auto c = Plans::MakeStagePlan(PlanStage(other, 0x52));
    merge = {};
    set = Plans::MergeStagePlans(set.get(), *PlanAnswer(shape, {c}, 2), merge);
    Require(set->stages[0].plans.size() == 2 && set->stages[0].layouts.size() == 1 && set->draws.size() == 2 && merge.added == 1 && merge.drawsAdded == 1, "a build with another key did not add a stage plan and a draw plan beside the first in the same layout");
    const std::array<const Plans::StagePlan*, 1> withFirst{first};
    const std::array<const Plans::StagePlan*, 1> withOther{set->stages[0].plans.front().get()};
    Require(Plans::FindDrawPlan(*set, withFirst) != nullptr && Plans::FindDrawPlan(*set, withOther) != nullptr && Plans::FindDrawPlan(*set, withFirst) != Plans::FindDrawPlan(*set, withOther), "each key's stage plan did not lead to its own draw plan");
    const auto recompiled = Plans::MakeStagePlan(PlanStage(PlanDraw{}, 0x53));
    merge = {};
    set = Plans::MergeStagePlans(set.get(), *PlanAnswer(shape, {recompiled}, 3), merge);
    Require(merge.replaced == 1 && merge.drawsEvicted == 1 && merge.drawsAdded == 1 && set->stages[0].plans.size() == 2 && Plans::FindDrawPlan(*set, withFirst) == nullptr, "a build whose key matched a stage plan with another compiled variant did not replace it and drop its draw plan");
    for (std::uint32_t index = 0; index < Plans::StagePlanLimits::PlansPerStage + 4; ++index) {
        PlanDraw many{};
        many.format = 0x01000000u + index;
        merge = {};
        set = Plans::MergeStagePlans(set.get(), *PlanAnswer(shape, {Plans::MakeStagePlan(PlanStage(many, 0x60 + index))}, 10 + index), merge);
    }
    Require(set->stages[0].plans.size() == Plans::StagePlanLimits::PlansPerStage && set->draws.size() <= Plans::StagePlanLimits::DrawPlans, "a stage holds " + std::to_string(set->stages[0].plans.size()) + " stage plans, past its cap");
    for (const auto& draw : set->draws) {
        Require(std::find(set->stages[0].plans.begin(), set->stages[0].plans.end(), draw.stages[0]) != set->stages[0].plans.end(), "a draw plan outlived its evicted stage plan");
    }
    std::size_t bytes = sizeof(Plans::StagePlanSet);
    for (const auto& plan : set->stages[0].plans) bytes += plan->bytes;
    Require(set->bytes >= bytes, "a stage plan set does not count the bytes of its plans");
    auto stranger = std::make_shared<LeanShape>();
    merge = {};
    const auto fresh = Plans::MergeStagePlans(set.get(), *PlanAnswer(stranger, {a}, 1), merge);
    Require(fresh->stages[0].plans.size() == 1 && fresh->draws.size() == 1 && fresh->shape == stranger, "a build for another shape kept the plans of the set it replaces");
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

namespace StagePlanReport = AgcDriver::DriverDetail::StagePlans;
using PlanCounter = StagePlanReport::Counter;

struct PlanTotals {
    std::array<std::uint64_t, static_cast<std::size_t>(PlanCounter::Count)> counters{};
    std::uint64_t differences = 0;
    std::uint64_t checked = 0;

    static PlanTotals Now() {
        PlanTotals totals;
        for (std::size_t index = 0; index < totals.counters.size(); ++index) totals.counters[index] = StagePlanReport::Total(static_cast<PlanCounter>(index));
        totals.differences = StagePlanReport::Differences();
        totals.checked = StagePlanReport::Checked(DrawRecords::Item::Fetch);
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
    return text;
}

void DevicePlans(bool ring, const std::array<std::byte, 1024>& todayColors) {
    using AgcDriver::Graphics::RuntimeSwitch;
    namespace RuntimeSwitches = AgcDriver::Graphics::RuntimeSwitches;
    const std::string name = " (lean front with stage plans)";
    RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, true);
    RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, false);
    RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, true);
    ColorMemory.fill(std::byte{0});
    const auto base = PlanTotals::Now();
    const auto recordsBase = Totals::Now();
    for (std::uint32_t submission = 0; submission < Submissions; ++submission) Submit(submission & 1u, 3, DrawsPerSubmission, name);
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(ColorMemory.data()), ColorMemory.size());
    Require(ColorMemory == todayColors, "the draws rendered differently from today's front" + name);
    const auto now = PlanTotals::Now();
    Require(now.differences == base.differences, "the stage plan verify found differences" + name + DescribePlans(now, base));
    Require(Totals::Now().Since(recordsBase, Counter::Draws) == 0, "draws were counted for draw records while stage plans replace them" + name);
    const auto draws = now.Since(base, PlanCounter::Draws);
    const auto off = now.Since(base, PlanCounter::OffTodaysDraw, PlanCounter::OffInline);
    if (!ring) {
        Require(now.Since(base, PlanCounter::Hits) == 0 && off == draws, "draws without the finisher ring were not counted as draws without stage plans" + name + DescribePlans(now, base));
        RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, false);
        return;
    }
    Require(draws == Submissions * DrawsPerSubmission, "stage plans counted " + std::to_string(draws) + " draws, expected " + std::to_string(Submissions * DrawsPerSubmission) + name + DescribePlans(now, base));
    const auto hits = now.Since(base, PlanCounter::Hits);
    const auto outcomes = hits + now.Since(base, PlanCounter::MissNoPlans, PlanCounter::MissFetchError) + off;
    Require(outcomes == draws, "the hits, misses and draws without stage plans add up to " + std::to_string(outcomes) + " of " + std::to_string(draws) + " draws" + name + DescribePlans(now, base));
    const auto jobs = draws - off;
    Require(jobs > DrawsPerSubmission && hits + DrawsPerSubmission >= jobs, "stage plans hit " + std::to_string(hits) + " of " + std::to_string(jobs) + " finisher jobs of one shape whose user data pointer moves between submissions" + name + DescribePlans(now, base));
    Require(now.Since(base, PlanCounter::DrawPlansAdded) == 1 && now.Since(base, PlanCounter::StagePlansAdded) == 2 && now.Since(base, PlanCounter::MissNewLayout, PlanCounter::MissNewCombination) == 0 && now.Since(base, PlanCounter::MissContentDiffers) == 0, "one shape whose constants move did not keep one draw plan of two stage plans that every later draw hits" + name + DescribePlans(now, base));
    const auto verified = now.Since(base, PlanCounter::HitsOnTodaysPath);
    const bool verifiedAsDue = StagePlanReport::VerifyEvery() == 0 ? verified == 0 : StagePlanReport::VerifyEvery() == 1 ? verified == hits : verified != 0 && verified < hits;
    Require(verifiedAsDue && (StagePlanReport::VerifyEvery() == 0 || now.checked > base.checked), "the stage plan verify did not run on the hits it was due for" + name + DescribePlans(now, base));
    const auto other = PlanTotals::Now();
    Submit(0, 6, 2, name);
    Submit(1, 6, 2, name);
    Submit(0, 9, 2, name);
    const auto counted = PlanTotals::Now();
    Require(counted.Since(other, PlanCounter::Hits) >= 1 && counted.Since(other, PlanCounter::Builds) == 0 && counted.Since(other, PlanCounter::MissNoPlans, PlanCounter::MissFetchError) == 0, "draws with other vertex counts, which are other draw identities, did not hit the shape's stage plans without a build" + name + DescribePlans(counted, other));
    RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, false);
    Submit(0, 3, 2, name);
    const auto off0 = PlanTotals::Now();
    Require(off0.Since(counted, PlanCounter::Draws) == 0 && off0.Since(counted, PlanCounter::ModeChanges) == 1 && off0.Since(counted, PlanCounter::SetsCleared) >= 1, "turning stage plans off did not clear the plans and stop counting draws" + name + DescribePlans(off0, counted));
    RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, true);
    Submit(0, 3, 2, name);
    const auto on = PlanTotals::Now();
    Require(on.Since(off0, PlanCounter::ModeChanges) == 1 && on.Since(off0, PlanCounter::MissNoPlans) >= 1 && on.Since(off0, PlanCounter::Hits) == 0, "turning stage plans on again did not start without plans" + name + DescribePlans(on, off0));
    Require(on.differences == base.differences, "the stage plan verify found differences after the switch flips" + name);
    StagePlanReport::Report(true);
    RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, false);
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
    const bool plansEnvironment = RuntimeSwitches::Set(RuntimeSwitch::StagePlans);
    Registered(VertexHeader, VertexCode.data(), sizeof(VertexCode), 2);
    Registered(PixelHeader, PixelCode.data(), sizeof(PixelCode), 1);
    for (auto& table : PixelConstants) table[0] = 0.25f;
    std::array<std::byte, 1024> todayColors{};
    RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, false);
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
            if (static_cast<Counter>(counter) == Counter::InputRefetches || static_cast<Counter>(counter) == Counter::HitsOnTodaysPath || static_cast<Counter>(counter) == Counter::HitsBaseValue || static_cast<Counter>(counter) == Counter::HitsBuffersPatched) continue;
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
    DevicePlans(ring, todayColors);
    RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, leanEnvironment);
    RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, recordsEnvironment);
    RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, plansEnvironment);
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
        Probes();
        Twins();
        StagePlanKeys();
        StagePlanMerges();
        StagePlanBuilds();
        const auto device = Device();
        std::cout << "draw record keys, store bounds, eviction, invalidation, variant probes, twins, stage plan keys, stage plan merges and stage plan build pacing passed" << (device == 0 ? "; repeated lean draws hit their identity records and their shape's stage plans" : "; the device part was skipped") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
