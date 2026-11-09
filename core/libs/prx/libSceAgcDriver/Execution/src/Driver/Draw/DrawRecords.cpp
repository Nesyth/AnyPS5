#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRecords.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/LeanDraw.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <type_traits>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::uint64_t FnvPrime = 0x100000001b3ull;

std::uint64_t mixWord(std::uint64_t key, std::uint64_t value) {
    key ^= value + 0x9e3779b97f4a7c15ull + (key << 6u) + (key >> 2u);
    key *= 0xff51afd7ed558ccdull;
    return key ^ (key >> 33u);
}

std::string hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

std::string keyText(const DrawRecordKey& key) {
    return "shape " + hex(key.shape) + " index " + hex(key.indexAddress) + "+" + std::to_string(key.indexCount) + (key.indexed ? " indexed" : " auto") + " instances " + std::to_string(key.instanceCount) + " first vertex " + std::to_string(key.firstVertex) + " first instance " + std::to_string(key.firstInstance);
}

}

DrawRecordKey DrawRecordKey::For(std::uint64_t shape, const Pm4::DrawParameters& parameters) {
    DrawRecordKey key;
    key.shape = shape;
    key.indexAddress = parameters.indexAddress;
    key.indexCount = parameters.indexCount;
    key.indexSize = parameters.indexSize;
    key.instanceCount = parameters.instanceCount;
    key.flags = parameters.flags;
    key.firstVertex = parameters.firstVertex;
    key.firstInstance = parameters.firstInstance;
    key.indexed = parameters.indexed;
    return key;
}

std::uint64_t DrawRecordKey::Probe() const {
    std::uint64_t probe = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) { probe = (probe ^ value) * FnvPrime; };
    mix(shape);
    mix(indexAddress);
    mix((static_cast<std::uint64_t>(indexCount) << 32u) | indexSize);
    mix((static_cast<std::uint64_t>(instanceCount) << 32u) | flags);
    mix((static_cast<std::uint64_t>(firstVertex) << 32u) | firstInstance);
    mix(indexed ? 1u : 0u);
    return probe == 0 ? 1 : probe;
}

DrawRecordHash HashDrawRecordWords(std::span<const std::uint32_t> first, std::span<const std::uint32_t> second) {
    std::uint64_t low = 0xcbf29ce484222325ull;
    std::uint64_t high = 0x84222325cbf29ce4ull ^ first.size();
    const auto visit = [&](std::span<const std::uint32_t> words) {
        for (const auto word : words) {
            low = (low ^ word) * FnvPrime;
            high = mixWord(high, word);
        }
    };
    visit(first);
    low = (low ^ 0xfffffffffull) * FnvPrime;
    high = mixWord(high, second.size() | 0x100000000ull);
    visit(second);
    return {low, high};
}

void DrawRecordSink::Reset() {
    queued = false;
    plan = false;
    verify = false;
    built = nullptr;
    shape = 0;
    contentKey.clear();
    vertexLayout.clear();
    vertexInputs.clear();
    fragmentOutputs.reset();
    writes.clear();
}

void DrawRecordSink::NoteVertices(std::span<const std::optional<ShaderRecompiler::ShaderVertexStageInfo>> infos) {
    vertexLayout.clear();
    vertexInputs.clear();
    for (const auto& info : infos) {
        if (!info) {
            vertexLayout.push_back(0);
            continue;
        }
        const auto count = std::min<std::uint32_t>(info->resourcesNum, ShaderRecompiler::ShaderVertexStageInfo::MaxResources);
        vertexLayout.insert(vertexLayout.end(), {0x80000000u | count, info->fetchAttribReg, info->fetchBufferReg, info->fetchEmbedded ? 1u : 0u, info->paClVsOutCntl, info->depthRemapMode, info->pointList ? 1u : 0u, std::bit_cast<std::uint32_t>(info->pointSize), std::bit_cast<std::uint32_t>(info->pointSizeMin), std::bit_cast<std::uint32_t>(info->pointSizeMax)});
        for (std::uint32_t index = 0; index < count; ++index) {
            const auto& fields = info->resources[index].fields;
            const auto& destination = info->resourcesDst[index];
            vertexLayout.insert(vertexLayout.end(), {fields[1] & 0xffff0000u, fields[3], static_cast<std::uint32_t>(destination.registerStart), static_cast<std::uint32_t>(destination.registersNum), static_cast<std::uint32_t>(destination.attrId), destination.fetchIndex});
            vertexInputs.insert(vertexInputs.end(), {fields[0], fields[1] & 0xffffu, fields[2]});
        }
    }
}

std::shared_ptr<DrawRecord> DrawRecord::Build(const DrawRecordKey& key, std::shared_ptr<const LeanShape> shape, std::uint64_t deviceSerial, std::uint64_t generation, std::uint64_t present, const DrawRecordSink& sink) {
    auto record = std::make_shared<DrawRecord>();
    record->key = key;
    record->probe = key.Probe();
    record->shape = std::move(shape);
    record->deviceSerial = deviceSerial;
    record->generation = generation;
    record->builtPresent = present;
    record->content = sink.Content();
    record->inputs = sink.Inputs();
    record->contentKey = sink.contentKey;
    record->vertexLayout = sink.vertexLayout;
    record->vertexInputs = sink.vertexInputs;
    record->fragmentOutputs = sink.fragmentOutputs;
    record->writes = sink.writes;
    record->plan = sink.built;
    record->signature = sink.built != nullptr ? sink.built->signature : 0;
    return record;
}

std::size_t DrawRecord::Bytes() const {
    return sizeof(DrawRecord) + 4 * (contentKey.capacity() + vertexLayout.capacity() + vertexInputs.capacity()) + sizeof(DrawPipeline::Range) * writes.capacity() + (plan != nullptr ? plan->bytes : 0);
}

const LeanShape* DrawRecordSet::Shape() const {
    return records.empty() ? nullptr : records.front()->shape.get();
}

const DrawRecord* DrawRecordSet::Match(const DrawRecordHash& content) const {
    for (const auto& record : records) {
        if (record->content == content) return record.get();
    }
    return nullptr;
}

std::size_t DrawRecordStore::SetOf(std::uint64_t probe) {
    return static_cast<std::size_t>((probe ^ (probe >> 31u)) % Sets) * Ways;
}

DrawRecordStore::Entry* DrawRecordStore::locate(const DrawRecordKey& key, std::uint64_t probe) {
    if (entries.empty()) return nullptr;
    const auto set = SetOf(probe);
    for (std::size_t way = 0; way < Ways; ++way) {
        auto& entry = entries[set + way];
        if (entry.used && entry.probe == probe && entry.key == key) return &entry;
    }
    return nullptr;
}

DrawRecordStore::Entry* DrawRecordStore::Find(const DrawRecordKey& key, std::uint64_t probe, std::uint64_t present) {
    auto* entry = locate(key, probe);
    if (entry == nullptr) return nullptr;
    entry->lastUse = ++uses;
    entry->lastPresent = present;
    return entry;
}

DrawRecordStore::Entry& DrawRecordStore::Place(const DrawRecordKey& key, std::uint64_t probe, std::uint64_t present, std::uint64_t front) {
    if (entries.empty()) entries.resize(Sets * Ways);
    const auto set = SetOf(probe);
    const auto aged = [&](const Entry& entry) { return present > entry.lastPresent + KeepPresents; };
    const auto before = [&](const Entry& entry, const Entry& other) {
        if (aged(entry) != aged(other)) return aged(entry);
        if ((entry.records == nullptr) != (other.records == nullptr)) return entry.records == nullptr;
        return entry.lastUse < other.lastUse;
    };
    Entry* victim = nullptr;
    for (std::size_t way = 0; way < Ways; ++way) {
        auto& entry = entries[set + way];
        if (!entry.used) {
            victim = &entry;
            break;
        }
        if (victim == nullptr || before(entry, *victim)) victim = &entry;
    }
    if (victim->used) {
        if (victim->records != nullptr) DrawRecords::AddOnWorker(DrawRecords::Counter::IdentitiesEvicted);
        Retire(*victim, front);
        --identities;
    }
    *victim = Entry{};
    victim->used = true;
    victim->key = key;
    victim->probe = probe;
    victim->lastUse = ++uses;
    victim->lastPresent = present;
    ++identities;
    noteGauges();
    return *victim;
}

DrawRecordStore::Installed DrawRecordStore::Install(const std::shared_ptr<const DrawRecord>& record, std::uint64_t present, std::uint64_t front) {
    auto* entry = locate(record->key, record->probe);
    if (entry == nullptr) return Installed::NoEntry;
    entry->buildSeq = 0;
    if (entry->refused) return Installed::Refused;
    auto set = std::make_shared<DrawRecordSet>();
    set->records.reserve(Variants);
    set->records.push_back(record);
    set->bytes = record->Bytes();
    if (entry->records != nullptr) {
        for (const auto& kept : entry->records->records) {
            if (set->records.size() >= Variants) break;
            if ((kept->content == record->content && kept->signature == record->signature) || kept->shape != record->shape) continue;
            set->records.push_back(kept);
            set->bytes += kept->Bytes();
        }
    }
    const auto held = [&] { return entry->records != nullptr ? entry->records->bytes : std::size_t{0}; };
    if (bytes - held() + set->bytes > byteCap) {
        sweep(present, front, entry);
        if (bytes - held() + set->bytes > byteCap) return Installed::OverCap;
    }
    Retire(*entry, front);
    records += set->records.size();
    bytes += set->bytes;
    entry->records = std::move(set);
    noteGauges();
    return Installed::Yes;
}

bool DrawRecordStore::Demote(const DrawRecordKey& key, std::uint64_t probe, std::uint64_t front) {
    auto* entry = locate(key, probe);
    if (entry == nullptr) return false;
    Retire(*entry, front);
    entry->refused = true;
    entry->buildSeq = 0;
    noteGauges();
    return true;
}

void DrawRecordStore::Retire(Entry& entry, std::uint64_t front) {
    if (entry.records == nullptr) return;
    records -= entry.records->records.size();
    bytes -= entry.records->bytes;
    if (entry.lastSeq >= front) retired.emplace_back(std::move(entry.records), entry.lastSeq);
    entry.records = nullptr;
}

void DrawRecordStore::sweep(std::uint64_t present, std::uint64_t front, const Entry* keep) {
    if (sweptPresent == present) return;
    sweptPresent = present;
    for (auto& entry : entries) {
        if (!entry.used || &entry == keep || present <= entry.lastPresent + KeepPresents) continue;
        if (entry.records != nullptr) DrawRecords::AddOnWorker(DrawRecords::Counter::IdentitiesEvicted);
        Retire(entry, front);
        entry = Entry{};
        --identities;
    }
}

void DrawRecordStore::Clear(std::uint64_t front) {
    if (records != 0) DrawRecords::AddOnWorker(DrawRecords::Counter::RecordsCleared, records);
    for (auto& entry : entries) {
        if (!entry.used) continue;
        Retire(entry, front);
        entry = Entry{};
    }
    identities = 0;
    records = 0;
    bytes = 0;
    sweptPresent = ~std::uint64_t{0};
    noteGauges();
}

void DrawRecordStore::Release(std::uint64_t front) {
    if (!retired.empty()) std::erase_if(retired, [&](const auto& entry) { return entry.second < front; });
}

void DrawRecordStore::Reset() {
    if (records != 0) DrawRecords::AddOnWorker(DrawRecords::Counter::RecordsCleared, records);
    std::vector<Entry>().swap(entries);
    retired.clear();
    identities = 0;
    records = 0;
    bytes = 0;
    uses = 0;
    sweptPresent = ~std::uint64_t{0};
    noteGauges();
}

void DrawRecordStore::noteGauges() const {
    DrawRecords::NoteStore(identities, records, bytes);
}

namespace DrawRecords {

namespace {

constexpr std::size_t FieldCount = static_cast<std::size_t>(Field::Count);
constexpr std::size_t AllowedKinds = static_cast<std::size_t>(Allowed::Count);
constexpr std::size_t CounterCount = static_cast<std::size_t>(Counter::Count);
constexpr std::uint64_t PrintedPerWindow = 8;
constexpr std::int64_t WindowMs = 10000;

struct FieldInfo {
    Item item;
    const char* name;
};

constexpr std::array<const char*, ItemCount> ItemNames{"lookup", "fetch", "commit"};

constexpr std::array<FieldInfo, FieldCount> FieldInfos{{
    {Item::Lookup, "keys"},
    {Item::Lookup, "shapes"},
    {Item::Lookup, "devices"},
    {Item::Lookup, "generations"},
    {Item::Fetch, "content keys"},
    {Item::Fetch, "vertex layouts"},
    {Item::Fetch, "write templates"},
    {Item::Fetch, "fragment outputs"},
    {Item::Fetch, "variants"},
    {Item::Fetch, "push constants"},
    {Item::Fetch, "binding words"},
    {Item::Fetch, "vertex attributes"},
    {Item::Fetch, "image views"},
    {Item::Fetch, "draw parameters"},
    {Item::Fetch, "read ranges"},
    {Item::Fetch, "snapshot words"},
}};

constexpr std::array<const char*, AllowedKinds> AllowedNames{"fragment outputs known on one side only", "today's draw not queued"};

constexpr std::array<const char*, CounterCount> CounterNames{
    "draws", "hits", "input refetches", "hits on today's path", "new identity", "second sighting", "build pending", "shape replaced", "refused",
    "content differs", "generation moved", "not queued", "keyless", "no fetch plan", "user word moved", "fetch gate", "image views moved", "vertex layout differs", "moved buffer fields", "memory changed", "fetch error",
    "today's draw", "indirect", "lean front verify", "inline",
    "records built", "fetch plans built", "fetch plans refused", "builds refused", "records installed", "builds dropped", "identities evicted", "records cleared", "record mode changes", "demotions",
};

static_assert(ItemNames.back() != nullptr && FieldInfos.back().name != nullptr && AllowedNames.back() != nullptr && CounterNames.back() != nullptr, "every draw record item, field, by-design case and counter needs a name");

struct Tally {
    std::atomic<std::uint64_t> value{0};
    std::uint64_t reported = 0;

    std::uint64_t Window() {
        const auto now = value.load(std::memory_order_relaxed);
        const auto since = now - reported;
        reported = now;
        return since;
    }
};

std::array<Tally, ItemCount> checked{};
std::array<Tally, FieldCount> differences{};
std::array<Tally, AllowedKinds> allowed{};
std::array<Tally, CounterCount> counters{};
std::atomic<std::uint64_t> printed{0};
std::atomic<std::uint64_t> storeIdentities{0};
std::atomic<std::uint64_t> storeRecords{0};
std::atomic<std::uint64_t> storeBytes{0};
std::mutex reportMutex;
std::int64_t reportedAt = 0;

std::uint64_t at(const std::array<std::uint64_t, CounterCount>& counts, Counter counter) {
    return counts[static_cast<std::size_t>(counter)];
}

std::string listed(std::size_t count, const auto& value, const auto& name) {
    std::string text;
    for (std::size_t index = 0; index < count; ++index) {
        if (value(index) == 0) continue;
        text += text.empty() ? std::string(": ") : std::string(", ");
        text += name(index);
        text += ' ';
        text += std::to_string(value(index));
    }
    return text;
}

std::string counted(const std::array<std::uint64_t, CounterCount>& counts, Counter first, Counter last) {
    std::string text;
    for (auto index = static_cast<std::size_t>(first); index <= static_cast<std::size_t>(last); ++index) {
        if (counts[index] == 0) continue;
        text += text.empty() ? "" : ", ";
        text += CounterNames[index];
        text += ' ';
        text += std::to_string(counts[index]);
    }
    return text.empty() ? std::string("none") : text;
}

}

namespace {

std::atomic<bool> corrupting{false};

}

void CorruptAnswers(bool corrupt) {
    corrupting.store(corrupt, std::memory_order_relaxed);
}

bool CorruptingAnswers() {
    return corrupting.load(std::memory_order_relaxed);
}

std::uint64_t VerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_DRAW_RECORDS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

void NoteChecked(Item item) {
    checked[static_cast<std::size_t>(item)].value.fetch_add(1, std::memory_order_relaxed);
}

void NoteDifference(Field field, std::string_view detail) {
    const auto& info = FieldInfos[static_cast<std::size_t>(field)];
    differences[static_cast<std::size_t>(field)].value.fetch_add(1, std::memory_order_relaxed);
    if (printed.fetch_add(1, std::memory_order_relaxed) < PrintedPerWindow) std::fprintf(stderr, "[draw-records-verify] %s %s differs: %.*s\n", ItemName(info.item), info.name, static_cast<int>(detail.size()), detail.data());
}

void NoteAllowed(Allowed kind) {
    allowed[static_cast<std::size_t>(kind)].value.fetch_add(1, std::memory_order_relaxed);
}

void Add(Counter counter, std::uint64_t count) {
    counters[static_cast<std::size_t>(counter)].value.fetch_add(count, std::memory_order_relaxed);
}

void AddOnWorker(Counter counter, std::uint64_t count) {
    auto& value = counters[static_cast<std::size_t>(counter)].value;
    value.store(value.load(std::memory_order_relaxed) + count, std::memory_order_relaxed);
}

void NoteStore(std::size_t identities, std::size_t records, std::size_t bytes) {
    storeIdentities.store(identities, std::memory_order_relaxed);
    storeRecords.store(records, std::memory_order_relaxed);
    storeBytes.store(bytes, std::memory_order_relaxed);
}

void Report(bool force) {
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    std::lock_guard lock(reportMutex);
    const auto last = reportedAt;
    if (last == 0) {
        reportedAt = now;
        if (!force) return;
    } else if (!force && now - last < WindowMs) {
        return;
    }
    reportedAt = now;
    const double seconds = last == 0 ? 0.0 : static_cast<double>(now - last) / 1000.0;
    std::array<std::uint64_t, CounterCount> counts{};
    for (std::size_t index = 0; index < CounterCount; ++index) counts[index] = counters[index].Window();
    std::array<std::uint64_t, ItemCount> checks{};
    std::uint64_t checksTotal = 0;
    for (std::size_t index = 0; index < ItemCount; ++index) checksTotal += checks[index] = checked[index].Window();
    std::array<std::uint64_t, FieldCount> differing{};
    std::uint64_t differingTotal = 0;
    for (std::size_t index = 0; index < FieldCount; ++index) differingTotal += differing[index] = differences[index].Window();
    std::array<std::uint64_t, AllowedKinds> byDesign{};
    std::uint64_t byDesignTotal = 0;
    for (std::size_t index = 0; index < AllowedKinds; ++index) byDesignTotal += byDesign[index] = allowed[index].Window();
    printed.store(0, std::memory_order_relaxed);
    const auto draws = at(counts, Counter::Draws);
    if (draws != 0 || at(counts, Counter::RecordModeChanges) != 0 || at(counts, Counter::RecordsCleared) != 0) {
        const auto hits = at(counts, Counter::Hits);
        AgcDriver::ProfilePrint_nid_no_patch("[draw-records] %.1f s: draws %llu, hits %llu (%.1f%%), input refetches %llu, hits on today's path %llu; misses: %s; records off: %s; store: %s; %llu identities, %llu records, %.1f KiB\n", seconds,
            static_cast<unsigned long long>(draws), static_cast<unsigned long long>(hits), draws != 0 ? 100.0 * static_cast<double>(hits) / static_cast<double>(draws) : 0.0,
            static_cast<unsigned long long>(at(counts, Counter::InputRefetches)), static_cast<unsigned long long>(at(counts, Counter::HitsOnTodaysPath)),
            counted(counts, Counter::MissNewIdentity, Counter::MissFetchError).c_str(), counted(counts, Counter::OffTodaysDraw, Counter::OffInline).c_str(), counted(counts, Counter::RecordsBuilt, Counter::Demotions).c_str(),
            static_cast<unsigned long long>(storeIdentities.load(std::memory_order_relaxed)), static_cast<unsigned long long>(storeRecords.load(std::memory_order_relaxed)), static_cast<double>(storeBytes.load(std::memory_order_relaxed)) / 1024.0);
    }
    if (VerifyEvery() == 0 || (draws == 0 && checksTotal == 0 && differingTotal == 0 && byDesignTotal == 0)) return;
    const auto fields = listed(FieldCount, [&](std::size_t index) { return differing[index]; }, [&](std::size_t index) { return std::string(ItemName(FieldInfos[index].item)) + " " + FieldInfos[index].name; });
    const auto cases = listed(AllowedKinds, [&](std::size_t index) { return byDesign[index]; }, [&](std::size_t index) { return std::string(AllowedNames[index]); });
    const auto check = [&](Item item) { return static_cast<unsigned long long>(checks[static_cast<std::size_t>(item)]); };
    AgcDriver::ProfilePrint_nid_no_patch("[draw-records-verify] %.1f s: every %llu; checked: lookups %llu, fetches %llu, commits %llu; differences %llu%s; by design %llu%s\n", seconds,
        static_cast<unsigned long long>(VerifyEvery()), check(Item::Lookup), check(Item::Fetch), check(Item::Commit), static_cast<unsigned long long>(differingTotal), fields.c_str(), static_cast<unsigned long long>(byDesignTotal), cases.c_str());
}

const char* ItemName(Item item) {
    return ItemNames[static_cast<std::size_t>(item)];
}

const char* FieldName(Field field) {
    return FieldInfos[static_cast<std::size_t>(field)].name;
}

std::uint64_t Checked(Item item) {
    return checked[static_cast<std::size_t>(item)].value.load(std::memory_order_relaxed);
}

std::uint64_t Differences() {
    std::uint64_t total = 0;
    for (const auto& field : differences) total += field.value.load(std::memory_order_relaxed);
    return total;
}

std::uint64_t Differences(Field field) {
    return differences[static_cast<std::size_t>(field)].value.load(std::memory_order_relaxed);
}

std::uint64_t AllowedCount(Allowed kind) {
    return allowed[static_cast<std::size_t>(kind)].value.load(std::memory_order_relaxed);
}

std::uint64_t Total(Counter counter) {
    return counters[static_cast<std::size_t>(counter)].value.load(std::memory_order_relaxed);
}

}

void Driver::probeDrawRecord(LeanDrawJob& job, const LeanShape& shape) {
    using DrawRecords::Counter;
    auto& store = leanDraws.identities;
    auto& finisher = DrawFinisher::Get();
    job.recordFlags = DrawRecordFlags::On;
    job.recordGeneration = frontLatch.generation;
    job.records = nullptr;
    DrawRecords::AddOnWorker(Counter::Draws);
    const auto key = DrawRecordKey::For(job.shapeKey, job.parameters);
    const auto probe = key.Probe();
    const auto present = Graphics::Recorder::Presents();
    const auto front = finisher.Front();
    const auto seq = finisher.Issued();
    auto* entry = store.Find(key, probe, present);
    if (entry == nullptr) {
        store.Place(key, probe, present, front);
        DrawRecords::AddOnWorker(Counter::MissNewIdentity);
        return;
    }
    if (entry->refused) {
        DrawRecords::AddOnWorker(Counter::MissRefused);
        return;
    }
    bool replaced = false;
    if (entry->records != nullptr && entry->records->Shape() != &shape) {
        store.Retire(*entry, front);
        replaced = true;
    }
    const bool pending = entry->BuildPending(front);
    if (entry->records == nullptr) {
        if (!pending) {
            job.recordFlags |= DrawRecordFlags::BuildWanted;
            entry->buildSeq = seq;
        }
        DrawRecords::AddOnWorker(replaced ? Counter::MissShapeReplaced : pending ? Counter::MissBuildPending : Counter::MissSecondSighting);
        return;
    }
    job.records = entry->records.get();
    job.recordFlags |= DrawRecordFlags::Probed;
    entry->lastSeq = seq;
    if (!pending) {
        job.recordFlags |= DrawRecordFlags::BuildWanted;
        entry->buildSeq = seq;
    }
    if (!DrawRecords::Due(frontLatch)) return;
    job.recordFlags |= DrawRecordFlags::VerifyDue;
    DrawRecords::NoteChecked(DrawRecords::Item::Lookup);
    bool differs = false;
    for (const auto& record : job.records->records) {
        if (!(record->key == key) || record->probe != probe) {
            DrawRecords::NoteDifference(DrawRecords::Field::Keys, "a record found for " + keyText(key) + " holds " + keyText(record->key));
            differs = true;
        }
        if (record->shape.get() != &shape) {
            DrawRecords::NoteDifference(DrawRecords::Field::Shapes, "a record for " + keyText(key) + " pins another shape than the draw's shape record built");
            differs = true;
        }
        if (record->deviceSerial != job.deviceSerial) {
            DrawRecords::NoteDifference(DrawRecords::Field::Devices, "a record for " + keyText(key) + " was built on device " + std::to_string(record->deviceSerial) + ", the draw runs on device " + std::to_string(job.deviceSerial));
            differs = true;
        }
        if (record->generation != job.recordGeneration) {
            DrawRecords::NoteDifference(DrawRecords::Field::Generations, "a record for " + keyText(key) + " was built at switch generation " + std::to_string(record->generation) + ", the segment latched generation " + std::to_string(job.recordGeneration));
            differs = true;
        }
    }
    if (!differs) return;
    store.Demote(key, probe, front);
    DrawRecords::AddOnWorker(Counter::Demotions);
    DrawRecords::AddOnWorker(Counter::MissRefused);
    job.records = nullptr;
    job.recordFlags = DrawRecordFlags::On;
}

namespace {

constexpr std::uint32_t NoBinding = ~0u;

std::uint64_t AlignedWord(std::uint32_t word) {
    return word & ~std::uint32_t{3};
}

std::uint32_t BufferTail(const std::uint32_t* words) {
    const auto stride = (words[1] >> 16u) & 0x3fffu;
    return (stride == 0u ? words[2] : stride * words[2]) & 3u;
}

bool BufferPresent(const std::uint32_t* words) {
    return words[2] != 0u && (words[0] != 0u || (words[1] & 0xffffu) != 0u);
}

bool SameBufferFields(const std::uint32_t* before, const std::uint32_t* after) {
    if (before[1] != after[1] || before[3] != after[3] || BufferPresent(before) != BufferPresent(after)) return false;
    if (!BufferPresent(after)) return true;
    return (before[0] & 3u) == (after[0] & 3u) && BufferTail(before) == BufferTail(after);
}

bool ReadRecordUserWords(const DrawProgram& program, const DrawUserWords& words, std::vector<std::uint32_t>& out) {
    const auto resources = program.nullPixel ? 0u : words.Value(program.resourceRegister);
    const auto count = ((resources >> 1u) & 0x1fu) | (((resources >> 27u) & 1u) << 5u);
    if (count > 32u) return false;
    out.clear();
    if (program.merged) out.assign(8, 0u);
    for (std::uint32_t index = 0; index < count; ++index) out.push_back(words.Value(program.userDataBase + index));
    if (!program.merged) return true;
    if (!program.mergedPointerRequired && !words.Contains(program.mergedPointer) && !words.Contains(program.mergedPointer + 1)) return true;
    out[0] = words.Value(program.mergedPointer);
    out[1] = words.Value(program.mergedPointer + 1);
    return !program.mergedPointerRequired || out[0] != 0u || out[1] != 0u;
}

bool SameVertexLayout(const ShaderRecompiler::ShaderVertexStageInfo& kept, const ShaderRecompiler::ShaderVertexStageInfo& now, bool& moved) {
    if (kept.resourcesNum != now.resourcesNum || kept.fetchAttribReg != now.fetchAttribReg || kept.fetchBufferReg != now.fetchBufferReg || kept.fetchEmbedded != now.fetchEmbedded || kept.paClVsOutCntl != now.paClVsOutCntl || kept.depthRemapMode != now.depthRemapMode || kept.pointList != now.pointList) return false;
    if (std::bit_cast<std::uint32_t>(kept.pointSize) != std::bit_cast<std::uint32_t>(now.pointSize) || std::bit_cast<std::uint32_t>(kept.pointSizeMin) != std::bit_cast<std::uint32_t>(now.pointSizeMin) || std::bit_cast<std::uint32_t>(kept.pointSizeMax) != std::bit_cast<std::uint32_t>(now.pointSizeMax)) return false;
    const auto count = std::min<std::uint32_t>(kept.resourcesNum, ShaderRecompiler::ShaderVertexStageInfo::MaxResources);
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto& a = kept.resources[index].fields;
        const auto& b = now.resources[index].fields;
        if ((a[1] & 0xffff0000u) != (b[1] & 0xffff0000u) || a[3] != b[3]) return false;
        const auto& from = kept.resourcesDst[index];
        const auto& to = now.resourcesDst[index];
        if (from.registerStart != to.registerStart || from.registersNum != to.registersNum || from.attrId != to.attrId || from.fetchIndex != to.fetchIndex) return false;
        if (a[0] != b[0] || (a[1] & 0xffffu) != (b[1] & 0xffffu) || a[2] != b[2]) moved = true;
    }
    return true;
}

struct FetchScratch {
    std::vector<std::vector<std::uint32_t>> userData;
    std::vector<std::vector<std::uint32_t>> liveWords;
    std::vector<std::vector<Graphics::DecodeRead>> decodeReads;
    std::vector<ShaderRecompiler::MemoryRegion> regions;
    std::vector<ShaderRecompiler::MemoryRegion> memory;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> live;
    DispatchVariant moved;
};

std::string drawText(const LeanDrawJob& job, const DrawRecordKey& key) {
    return "draw at submission " + std::to_string(job.serial) + " DWORD " + std::to_string(job.packetOffset) + " (" + keyText(key) + ")";
}

template <typename T>
std::string firstDifference(const std::vector<T>& record, const std::vector<T>& today) {
    if (record.size() != today.size()) return std::to_string(record.size()) + " words in the record, " + std::to_string(today.size()) + " today";
    for (std::size_t index = 0; index < record.size(); ++index) {
        if (record[index] == today[index]) continue;
        if constexpr (std::is_integral_v<T>) return "word " + std::to_string(index) + " is " + hex(record[index]) + " in the record, " + hex(today[index]) + " today";
        else return "entry " + std::to_string(index) + " differs";
    }
    return "equal";
}

std::string bindingDifference(const std::vector<std::uint32_t>& record, const std::vector<std::uint32_t>& today) {
    constexpr std::size_t HeaderWords = 6;
    std::size_t at = 0;
    for (std::size_t binding = 0; at + HeaderWords <= record.size() && at + HeaderWords <= today.size(); ++binding) {
        const auto words = record[at + HeaderWords - 1];
        if (!std::equal(record.begin() + static_cast<std::ptrdiff_t>(at), record.begin() + static_cast<std::ptrdiff_t>(at + HeaderWords), today.begin() + static_cast<std::ptrdiff_t>(at))) return "binding " + std::to_string(binding) + " (role " + std::to_string(record[at + 1]) + ", kind " + std::to_string(record[at]) + ") has another shape: " + std::to_string(words) + " words in the record, " + std::to_string(today[at + HeaderWords - 1]) + " today";
        for (std::size_t word = 0; word < words && at + HeaderWords + word < record.size(); ++word) {
            const auto kept = record[at + HeaderWords + word];
            const auto now = today[at + HeaderWords + word];
            if (kept != now) return "binding " + std::to_string(binding) + " (role " + std::to_string(record[at + 1]) + ", kind " + std::to_string(record[at]) + ", set " + std::to_string(record[at + 2]) + ", binding " + std::to_string(record[at + 3]) + ") word " + std::to_string(word) + " is " + hex(kept) + " in the record, " + hex(now) + " today";
        }
        at += HeaderWords + words;
    }
    return firstDifference(record, today);
}

}

void DrawRecordView::Capture(std::span<const Graphics::CompiledShader> shaders, const Pm4::DrawParameters& drawParameters) {
    stages.clear();
    for (const auto& shader : shaders) {
        auto& view = stages.emplace_back();
        if (shader.program == nullptr) continue;
        view.variant = shader.program->variantId;
        view.stage = static_cast<std::uint32_t>(shader.stage);
        view.pushOffset = shader.pushConstantOffset;
        for (const auto& binding : shader.program->bindings) {
            const auto& words = Graphics::DrawGuestWords(shader, binding);
            view.bindings.insert(view.bindings.end(), {static_cast<std::uint32_t>(binding.kind), static_cast<std::uint32_t>(binding.role), binding.descriptorSet, binding.binding, binding.count, static_cast<std::uint32_t>(words.size())});
            view.bindings.insert(view.bindings.end(), words.begin(), words.end());
        }
        for (const auto& attribute : Graphics::DrawVertexAttributes(shader)) {
            const auto& fields = attribute.resource.fields;
            view.vertexAttributes.insert(view.vertexAttributes.end(), {attribute.location, attribute.components, fields[0], fields[1], fields[2], fields[3], attribute.fetchIndex, attribute.formatComponents});
        }
        if (shader.registeredImageViews != nullptr) view.views = *shader.registeredImageViews;
    }
    pushed = false;
    try {
        push = Graphics::AssemblePushConstants(shaders);
        pushed = true;
    } catch (const std::exception&) {
        push = {};
    }
    parameters = {drawParameters.indexAddress, drawParameters.indexCount | static_cast<std::uint64_t>(drawParameters.indexSize) << 32u, drawParameters.instanceCount | static_cast<std::uint64_t>(drawParameters.flags) << 32u, drawParameters.firstVertex | static_cast<std::uint64_t>(drawParameters.firstInstance) << 32u, drawParameters.indexed ? 1u : 0u, drawParameters.indirect.has_value() ? 1u : 0u, 0, 0};
}

void DrawRecordView::CaptureSnapshots(std::span<const Graphics::GuestMemorySnapshot> snapshots) {
    snapshotWords.clear();
    for (const auto& snapshot : snapshots) {
        for (std::size_t at = 0; at + sizeof(std::uint32_t) <= snapshot.bytes.size(); at += sizeof(std::uint32_t)) {
            std::uint32_t word = 0;
            std::memcpy(&word, snapshot.bytes.data() + at, sizeof(word));
            snapshotWords.emplace_back(snapshot.address + at, word);
        }
    }
    std::sort(snapshotWords.begin(), snapshotWords.end());
    snapshotWords.erase(std::unique(snapshotWords.begin(), snapshotWords.end()), snapshotWords.end());
}

std::shared_ptr<const DrawRecordPlan> Driver::buildDrawRecordPlan(const Graphics::State& graphics, const Pm4::DrawParameters& parameters, const std::vector<DrawProgram>& programs, const std::vector<ShaderRecompiler::ProgramRole>& roles, const std::vector<StageCapture>& stageCaptures, const std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>& vertexInfos, const std::vector<std::vector<Graphics::DecodeRead>>& decodeReads, std::span<const Graphics::CompiledShader> stages) {
    using Role = ShaderRecompiler::ProgramRole;
    using DescriptorRole = ShaderRecompiler::DescriptorRole;
    if (graphics.rectList || graphics.stages.mesh || graphics.stages.tessellation || graphics.stages.tessellationGeometry || graphics.stages.path != Graphics::ShaderPath::Vertex || parameters.indirect) return nullptr;
    const auto count = programs.size();
    if (count == 0 || stages.size() != count || stageCaptures.size() != count || roles.size() != count || vertexInfos.size() != count || decodeReads.size() != count) return nullptr;
    auto plan = std::make_shared<DrawRecordPlan>();
    plan->stages.resize(count);
    std::uint64_t signature = 0xcbf29ce484222325ull;
    const auto sign = [&](std::uint64_t value) { signature = mixWord(signature, value); };
    for (std::size_t i = 0; i < count; ++i) {
        const auto& program = programs[i];
        const auto& capture = stageCaptures[i];
        if (roles[i] == Role::GeometryBack || capture.compiled == nullptr || capture.capture == nullptr || stages[i].program != capture.compiled.get() || stages[i].heap != nullptr) return nullptr;
        if (capture.direct || capture.ring || capture.heapWords != nullptr || program.snapshot == nullptr || program.snapshot->header.empty() || program.tailContinuation != nullptr) return nullptr;
        const auto& compiled = *capture.compiled;
        if (compiled.gpuResolvedImages || compiled.heapTemplateId != 0 || compiled.deviceImageTables || compiled.dataAddressPush || !CacheableResult(compiled)) return nullptr;
        if (std::ranges::any_of(compiled.bindings, [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == DescriptorRole::BdaPagetable; })) plan->addressed = true;
        auto& stage = plan->stages[i];
        stage.stage = stages[i].stage;
        stage.pushConstantOffset = stages[i].pushConstantOffset;
        stage.hiddenUserWords = program.hiddenUserWords;
        stage.compiled = capture.compiled;
        stage.registeredViews = capture.registeredViews;
        stage.userData = program.userData;
        stage.userDataMask = capture.userDataMask;
        stage.walkedWords = capture.userDataUse.asBase | capture.userDataUse.asValue;
        auto variant = std::make_shared<DispatchVariant>();
        variant->compiled = capture.compiled;
        variant->registeredViews = capture.registeredViews;
        variant->shader = program.snapshot;
        variant->forgetSerial = capture.forgetSerial;
        variant->pushOffset = capture.pushOffset;
        variant->userDataMask = capture.userDataMask;
        std::vector<ShaderRecompiler::MemoryRegion> regions(capture.regions.begin(), capture.regions.end());
        std::stable_sort(regions.begin(), regions.end(), [](const ShaderRecompiler::MemoryRegion& a, const ShaderRecompiler::MemoryRegion& b) { return a.guestAddress < b.guestAddress; });
        tagVariantRuns(*variant, regions, capture.userDataUse, true, program.hiddenUserWords, capture.supplied);
        tagUserValueWords(*variant, program, nullptr);
        if (!capture.capture->readTrace.leaves.empty()) recordStageDataPositions(*variant, *capture.capture, decodeReads[i]);
        stage.runBases = variant->runBases;
        stage.relocatableWords = variant->relocatableWords;
        stage.valueWords = variant->valueWords;
        stage.valueSlots = variant->valueSlots;
        stage.walkedWords |= variant->calleeWords;
        auto& targets = stage.targets;
        const auto byPosition = [](const DrawRecordTarget& a, const DrawRecordTarget& b) { return a.position < b.position || (a.position == b.position && (a.binding < b.binding || (a.binding == b.binding && a.word < b.word))); };
        std::uint32_t flatBinding = NoBinding;
        for (std::size_t b = 0; b < compiled.bindings.size() && flatBinding == NoBinding; ++b) {
            if (compiled.bindings[b].role == DescriptorRole::FlattenedSrt) flatBinding = static_cast<std::uint32_t>(b);
        }
        const bool flat = flatBinding != NoBinding && !compiled.bindings[flatBinding].guestDescriptor.empty();
        if (flat && variant->flatBinding == flatBinding) {
            const auto& slots = compiled.bindings[variant->flatBinding].guestDescriptor;
            for (std::size_t k = 0; k < variant->dataPositions.size() && k < variant->dataSlots.size(); ++k) {
                if (variant->dataSlots[k] < slots.size()) targets.push_back({variant->dataPositions[k], variant->flatBinding, variant->dataSlots[k]});
            }
        }
        std::sort(targets.begin(), targets.end(), byPosition);
        const auto flatTarget = [&](std::uint32_t position) { return std::any_of(targets.begin(), targets.end(), [&](const DrawRecordTarget& target) { return target.position == position; }); };
        const auto& words = variant->words;
        std::vector<std::pair<std::size_t, std::size_t>> runWords;
        for (std::size_t offset = 0; const auto& [begin, end] : variant->runs) {
            const auto length = static_cast<std::size_t>((end - begin) / sizeof(std::uint32_t));
            runWords.emplace_back(offset, length);
            offset += length;
        }
        const auto readThrough = [&](const std::uint32_t* descriptor) {
            const auto base = static_cast<std::uint64_t>(descriptor[0]) | static_cast<std::uint64_t>(descriptor[1] & 0xffffu) << 32u;
            const auto stride = (descriptor[1] >> 16u) & 0x3fffu;
            const auto bytes = std::max<std::uint64_t>(stride != 0 ? static_cast<std::uint64_t>(stride) * descriptor[2] : descriptor[2], 16u);
            return std::any_of(variant->runs.begin(), variant->runs.end(), [&](const auto& run) { return run.first < base + bytes && run.second > base; });
        };
        const auto inUserData = [&](const std::uint32_t* descriptor) {
            for (std::size_t index = 0; index + 4 <= program.userData.size(); ++index) {
                if (std::equal(descriptor, descriptor + 4, program.userData.begin() + static_cast<std::ptrdiff_t>(index))) return true;
            }
            return false;
        };
        const auto countIn = [](std::span<const std::uint32_t> values, std::uint32_t value) { return static_cast<std::size_t>(std::count(values.begin(), values.end(), value)); };
        const auto* flatSlots = flat ? &compiled.bindings[flatBinding].guestDescriptor : nullptr;
        bool explained = true;
        for (std::size_t slot = 0; flat && explained && slot < flatSlots->size(); ++slot) {
            const bool leaf = std::any_of(targets.begin(), targets.end(), [&](const DrawRecordTarget& target) { return target.binding == flatBinding && target.word == slot; });
            if (!leaf && countIn(words, (*flatSlots)[slot]) == 0) explained = false;
        }
        std::vector<std::uint32_t> pushWords(compiled.pushConstants.size() / sizeof(std::uint32_t));
        if (!pushWords.empty()) std::memcpy(pushWords.data(), compiled.pushConstants.data(), pushWords.size() * sizeof(std::uint32_t));
        std::vector<DrawRecordTarget> bufferTargets;
        for (std::size_t b = 0; explained && b < compiled.bindings.size(); ++b) {
            const auto& binding = compiled.bindings[b];
            if (binding.role != DescriptorRole::GuestBuffers || binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 4u) continue;
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                if ((element < binding.bufferWritten.size() && binding.bufferWritten[element]) || (element < binding.bufferAtomic.size() && binding.bufferAtomic[element])) continue;
                const auto word = element * 4u;
                if (std::any_of(stage.valueSlots.begin(), stage.valueSlots.end(), [&](const auto& slot) { return slot[1] == b && slot[2] / 4u == element; })) continue;
                const auto* descriptor = binding.guestDescriptor.data() + word;
                if (!BufferPresent(descriptor) || inUserData(descriptor) || readThrough(descriptor)) continue;
                std::optional<std::uint32_t> found;
                bool ambiguous = false;
                for (const auto& [offset, length] : runWords) {
                    for (std::size_t at = offset; at + 4 <= offset + length && !ambiguous; ++at) {
                        if (!std::equal(descriptor, descriptor + 4, words.begin() + static_cast<std::ptrdiff_t>(at))) continue;
                        if (found) ambiguous = true;
                        else found = static_cast<std::uint32_t>(at);
                    }
                }
                if (!found || ambiguous || flatTarget(*found)) continue;
                const auto base = descriptor[0];
                if (countIn(words, base) != 1 || countIn(pushWords, base) != 0 || std::find(program.userData.begin(), program.userData.end(), base) != program.userData.end()) continue;
                bool foreign = false;
                for (std::size_t other = 0; other < compiled.bindings.size() && !foreign; ++other) {
                    const auto& kept = compiled.bindings[other];
                    if (other == flatBinding || kept.role == DescriptorRole::GuestBuffers) continue;
                    foreign = countIn(kept.guestDescriptor, base) != 0;
                }
                if (foreign) continue;
                stage.buffers.push_back({static_cast<std::uint32_t>(b), element, *found});
                bufferTargets.push_back({*found, static_cast<std::uint32_t>(b), word});
                for (std::size_t slot = 0; flat && slot < flatSlots->size(); ++slot) {
                    if ((*flatSlots)[slot] == base) bufferTargets.push_back({*found, flatBinding, static_cast<std::uint32_t>(slot)});
                }
            }
        }
        for (std::size_t b = 0; b < compiled.bindings.size(); ++b) {
            const auto& kept = compiled.bindings[b];
            if (kept.role != DescriptorRole::GuestBuffers) continue;
            for (std::size_t word = 0; word < kept.guestDescriptor.size(); ++word) {
                const auto value = kept.guestDescriptor[word];
                const bool listed = std::any_of(bufferTargets.begin(), bufferTargets.end(), [&](const DrawRecordTarget& target) { return target.binding == b && target.word == word; });
                if (listed || std::none_of(bufferTargets.begin(), bufferTargets.end(), [&](const DrawRecordTarget& target) { return words[target.position] == value; })) continue;
                std::erase_if(bufferTargets, [&](const DrawRecordTarget& target) { return words[target.position] == value; });
                std::erase_if(stage.buffers, [&](const DrawRecordBuffer& buffer) { return words[buffer.position] == value; });
            }
        }
        targets.insert(targets.end(), bufferTargets.begin(), bufferTargets.end());
        std::sort(targets.begin(), targets.end(), byPosition);
        targets.erase(std::unique(targets.begin(), targets.end(), [](const DrawRecordTarget& a, const DrawRecordTarget& b) { return a.position == b.position && a.binding == b.binding && a.word == b.word; }), targets.end());
        variant->dataPositions.clear();
        variant->dataSlots.clear();
        for (const auto& target : targets) {
            if (variant->dataPositions.empty() || variant->dataPositions.back() != target.position) variant->dataPositions.push_back(target.position);
        }
        std::uint32_t shaderData = NoBinding;
        for (std::size_t b = 0; b < compiled.bindings.size(); ++b) {
            if (compiled.bindings[b].role == DescriptorRole::ShaderData) {
                shaderData = static_cast<std::uint32_t>(b);
                break;
            }
        }
        const auto& forwarded = compiled.shaderDataUserWords;
        std::uint64_t refused = 0;
        for (std::uint32_t slot = 0; slot < forwarded.size(); ++slot) {
            const auto index = forwarded[slot];
            if (index >= program.userData.size() || index >= 64u) continue;
            const auto value = program.userData[index];
            bool placed = false;
            if ((static_cast<std::size_t>(slot) + 1) * sizeof(std::uint32_t) <= compiled.pushConstants.size()) {
                std::uint32_t pushed = 0;
                std::memcpy(&pushed, compiled.pushConstants.data() + static_cast<std::size_t>(slot) * sizeof(std::uint32_t), sizeof(pushed));
                if (pushed == value) {
                    stage.forwards.push_back({index, slot, DrawRecordForward::Push});
                    placed = true;
                }
            }
            if (shaderData != NoBinding && slot < compiled.bindings[shaderData].guestDescriptor.size() && compiled.bindings[shaderData].guestDescriptor[slot] == value) {
                stage.forwards.push_back({index, slot, shaderData});
                placed = true;
            }
            if (!placed) refused |= std::uint64_t{1} << index;
            else stage.forwardedWords |= std::uint64_t{1} << index;
        }
        stage.forwardedWords &= ~refused;
        std::erase_if(stage.forwards, [&](const DrawRecordForward& forward) { return (refused >> forward.word & 1u) != 0; });
        if (vertexInfos[i]) {
            stage.vertexInfo = *vertexInfos[i];
            for (const auto reg : {vertexInfos[i]->fetchAttribReg, vertexInfos[i]->fetchBufferReg}) {
                if (reg + 1u < 64u) stage.vertexWords |= std::uint64_t{3} << reg;
            }
        }
        stage.regions.clear();
        appendEntryRegions(*variant, stage.regions);
        stage.variant = std::move(variant);
        const auto structural = stage.walkedWords & ~stage.relocatableWords & ~stage.valueWords;
        sign(i);
        sign(compiled.variantId);
        for (std::size_t index = 0; index < stage.userData.size(); ++index) {
            if (index < stage.hiddenUserWords || (index < 64u && (structural >> index & 1u) != 0)) sign((static_cast<std::uint64_t>(index) << 32u) | stage.userData[index]);
        }
        for (std::size_t run = 0; run < stage.variant->runs.size(); ++run) sign(run < stage.runBases.size() && stage.runBases[run] >= 0 ? static_cast<std::uint64_t>(stage.runBases[run]) << 48u | (stage.variant->runs[run].second - stage.variant->runs[run].first) : stage.variant->runs[run].first);
        for (std::size_t position = 0, target = 0; position < stage.variant->words.size(); ++position) {
            bool data = false;
            while (target < targets.size() && targets[target].position == position) {
                ++target;
                data = true;
            }
            if (!data) sign(stage.variant->words[position]);
        }
        plan->bytes += sizeof(DrawRecordStage) + sizeof(DispatchVariant) + sizeof(std::uint32_t) * (stage.variant->words.size() + stage.variant->dataPositions.size() + stage.userData.size()) + sizeof(std::pair<std::uint64_t, std::uint64_t>) * stage.variant->runs.size() + sizeof(ShaderRecompiler::MemoryRegion) * stage.regions.size() + sizeof(DrawRecordTarget) * targets.size() + sizeof(ShaderRecompiler::RecompileResult) + compiled.pushConstants.size();
        for (const auto& binding : compiled.bindings) plan->bytes += sizeof(binding) + sizeof(std::uint32_t) * binding.guestDescriptor.size();
    }
    plan->signature = signature == 0 ? 1 : signature;
    return plan;
}

DrawRecords::Counter Driver::fetchDrawRecord(const LeanDrawJob& job, const DrawRecord& record, DrawRecordHit& hit) {
    using DrawRecords::Counter;
    using DescriptorRole = ShaderRecompiler::DescriptorRole;
    const auto* plan = record.plan.get();
    if (plan == nullptr) return Counter::MissNoFetchPlan;
    if (record.generation != job.recordGeneration || record.deviceSerial != job.deviceSerial || record.shape.get() != job.shape) return Counter::MissGenerationMoved;
    const auto& decode = *job.shape->decode;
    const auto& programs = decode.programs;
    const auto count = programs.size();
    if (plan->stages.size() != count) return Counter::MissNoFetchPlan;
    struct ScratchStorage {};
    auto& scratch = HostThreadLocal<FetchScratch, ScratchStorage>();
    if (scratch.userData.size() < count) {
        scratch.userData.resize(count);
        scratch.liveWords.resize(count);
        scratch.decodeReads.resize(count);
    }
    scratch.memory.clear();
    hit.sides.assign(count, {});
    hit.words.assign(plan->addressed ? count : 0, {});
    hit.decodeReads.assign(plan->addressed ? count : 0, {});
    hit.snapshots.clear();
    hit.stages.clear();
    hit.stages.reserve(count);
    hit.inputsMoved = false;
    std::uint64_t imagesFlushed = 0, runsSynced = 0;
    std::optional<SampledReadScope> sampling;
    const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& stage = plan->stages[i];
        const auto& program = programs[i];
        const auto& compiled = *stage.compiled;
        auto& user = scratch.userData[i];
        if (!ReadRecordUserWords(program, job.words, user) || user.size() != stage.userData.size()) return Counter::MissUserWord;
        if (stage.registeredViews != nullptr && !CaptureRegisteredViewsCurrent(*stage.registeredViews)) return Counter::MissImageViews;
        bool relocated = false;
        for (std::size_t index = 0; index < user.size(); ++index) {
            const auto from = stage.userData[index];
            const auto to = user[index];
            if (from == to) continue;
            if (index < stage.hiddenUserWords || index >= 64u) return Counter::MissUserWord;
            const auto bit = std::uint64_t{1} << index;
            if ((stage.relocatableWords & bit) != 0) {
                if (from == 0 || to == 0 || ((from ^ to) & 3u) != 0) return Counter::MissUserWord;
                relocated = true;
            } else if ((stage.valueWords & bit) != 0) {
                if (from == 0 || to == 0) return Counter::MissUserWord;
            } else if ((stage.walkedWords & bit) != 0) {
                return Counter::MissUserWord;
            } else if ((stage.vertexWords & bit) == 0 && (stage.forwardedWords & bit) == 0 && (stage.userDataMask & bit) != 0) {
                return Counter::MissUserWord;
            }
        }
        const DispatchVariant* variant = stage.variant.get();
        const auto* regions = &stage.regions;
        if (relocated) {
            auto& moved = scratch.moved;
            moved.runs.assign(variant->runs.begin(), variant->runs.end());
            for (std::size_t run = 0; run < moved.runs.size() && run < stage.runBases.size(); ++run) {
                const auto base = stage.runBases[run];
                if (base < 0) continue;
                const auto delta = AlignedWord(user[static_cast<std::size_t>(base)]) - AlignedWord(stage.userData[static_cast<std::size_t>(base)]);
                moved.runs[run].first += delta;
                moved.runs[run].second += delta;
            }
            moved.dataPositions.assign(variant->dataPositions.begin(), variant->dataPositions.end());
            moved.compiled = variant->compiled;
            moved.shared.clear();
            variant = &moved;
            scratch.regions.clear();
            appendEntryRegions(moved, scratch.regions, &stage.variant->words);
            regions = &scratch.regions;
        }
        auto& live = scratch.live;
        auto outcome = validateVariant(program.binary.codeAddress, job.queue, *variant, *regions, imagesFlushed, runsSynced, sampling, &live);
        if (gateRetry() && (outcome == EntryOutcome::PublishMoved || outcome == EntryOutcome::PendingMoved)) outcome = validateVariant(program.binary.codeAddress, job.queue, *variant, *regions, imagesFlushed, runsSynced, sampling, &live);
        if (outcome == EntryOutcome::Differing) return Counter::MissContentDiffers;
        if (outcome != EntryOutcome::Equal && outcome != EntryOutcome::EqualData) return Counter::MissFetchGate;
        if (outcome == EntryOutcome::Equal) live.clear();
        auto& side = hit.sides[i];
        const auto patch = [&](std::uint32_t binding, std::uint32_t word, std::uint32_t value) {
            if (binding >= compiled.bindings.size()) return false;
            if (side.bindings.empty()) side.bindings.resize(compiled.bindings.size());
            auto& descriptor = side.bindings[binding];
            if (descriptor.empty()) descriptor = compiled.bindings[binding].guestDescriptor;
            if (word >= descriptor.size()) return false;
            descriptor[word] = value;
            return true;
        };
        auto& liveWords = plan->addressed ? hit.words[i] : scratch.liveWords[i];
        liveWords.assign(stage.variant->words.begin(), stage.variant->words.end());
        std::size_t target = 0;
        for (const auto& [position, value] : live) {
            while (target < stage.targets.size() && stage.targets[target].position < position) ++target;
            if (target == stage.targets.size() || stage.targets[target].position != position || position >= liveWords.size()) return Counter::MissFetchError;
            for (auto at = target; at < stage.targets.size() && stage.targets[at].position == position; ++at) {
                if (!patch(stage.targets[at].binding, stage.targets[at].word, value)) return Counter::MissFetchError;
            }
            liveWords[position] = value;
        }
        for (const auto& [index, binding, word] : stage.valueSlots) {
            if (index < user.size() && user[index] != stage.userData[index] && !patch(binding, word, user[index])) return Counter::MissFetchError;
        }
        for (const auto& forward : stage.forwards) {
            const auto value = user[forward.word];
            if (value == stage.userData[forward.word]) continue;
            if (forward.binding == DrawRecordForward::Push) side.pushWords.emplace_back(forward.slot, value);
            else if (!patch(forward.binding, forward.slot, value)) return Counter::MissFetchError;
        }
        for (std::size_t b = 0; b < side.bindings.size(); ++b) {
            const auto& kept = compiled.bindings[b];
            const auto& now = side.bindings[b];
            if (now.empty() || kept.role != DescriptorRole::GuestBuffers || now.size() != kept.guestDescriptor.size()) continue;
            for (std::size_t word = 0; word + 4 <= now.size(); word += 4) {
                if (!std::equal(now.begin() + static_cast<std::ptrdiff_t>(word), now.begin() + static_cast<std::ptrdiff_t>(word + 4), kept.guestDescriptor.begin() + static_cast<std::ptrdiff_t>(word)) && !SameBufferFields(kept.guestDescriptor.data() + word, now.data() + word)) return Counter::MissBufferFields;
            }
        }
        if (stage.vertexInfo) {
            auto& reads = plan->addressed ? hit.decodeReads[i] : scratch.decodeReads[i];
            auto info = DecodeOrderedVertexStageInfo(program.binary.header, program.binary.headerAddress, user, reads, 8u - program.firstUserSgpr);
            info.paClVsOutCntl = stage.vertexInfo->paClVsOutCntl;
            info.depthRemapMode = stage.vertexInfo->depthRemapMode;
            Graphics::ApplyPointSize(decode.state, info);
            bool moved = false;
            if (!SameVertexLayout(*stage.vertexInfo, info, moved)) return Counter::MissVertexLayout;
            if (moved && !compiled.vertexAttributes.empty()) {
                side.vertexAttributes = compiled.vertexAttributes;
                for (auto& attribute : side.vertexAttributes) {
                    if (attribute.location >= info.resourcesNum || attribute.location >= info.resources.size()) return Counter::MissVertexLayout;
                    attribute.resource = info.resources[attribute.location];
                    if (info.fetchEmbedded) attribute.fetchIndex = info.resourcesDst[attribute.location].fetchIndex;
                }
                hit.inputsMoved = true;
            }
            for (const auto& read : reads) scratch.memory.push_back({read.address, read.Bytes()});
        }
        const bool sided = !side.bindings.empty() || !side.pushWords.empty() || !side.vertexAttributes.empty();
        hit.stages.push_back({stage.stage, stage.compiled.get(), stage.pushConstantOffset, stage.registeredViews.get(), sided ? &side : nullptr});
        appendEntryRegions(*variant, scratch.memory, &liveWords);
    }
    AppendProgramMemory(scratch.memory, programs);
    hit.parameters = job.parameters;
    const auto& front = *plan->stages.front().compiled;
    const auto& firstUser = scratch.userData.front();
    const auto userWord = [&](std::int32_t sgpr, std::uint32_t& value) {
        if (sgpr < 0 || static_cast<std::uint32_t>(sgpr) < programs.front().firstUserSgpr) return false;
        const auto index = static_cast<std::uint32_t>(sgpr) - programs.front().firstUserSgpr;
        if (index >= firstUser.size()) return false;
        value = firstUser[index];
        return true;
    };
    std::uint32_t offset = 0;
    if (front.vertexOffsetSgpr >= 0) {
        if (!userWord(front.vertexOffsetSgpr, offset) || offset > std::numeric_limits<std::uint32_t>::max() - hit.parameters.firstVertex) return Counter::MissFetchError;
        hit.parameters.firstVertex += offset;
    }
    if (front.instanceOffsetSgpr >= 0) {
        if (!userWord(front.instanceOffsetSgpr, offset)) return Counter::MissFetchError;
        hit.parameters.firstInstance = offset;
    }
    runCaptureCheckHook();
    if (capturedMemoryChanged(scratch.memory) || recordQueuedLabelsAfterCapture(job.queue, scratch.memory)) return Counter::MissMemoryChanged;
    hit.reads = QueuedDrawReadRanges(hit.parameters, hit.stages);
    if (plan->addressed) {
        hit.snapshots.reserve(scratch.memory.size());
        for (const auto& region : scratch.memory) hit.snapshots.push_back({region.guestAddress, region.bytes});
    }
    return Counter::Hits;
}

std::shared_ptr<DrawRecordHit> Driver::fetchDrawRecords(const LeanDrawJob& job) {
    using DrawRecords::Counter;
    DrawPipeline::FollowEpoch(job.epoch);
    DrawPipeline::Active() = true;
    GuestMemory::SetCurrentPacket(job.opcode, job.queue);
    const DrawFinisher::AheadScope ahead;
    auto hit = std::make_shared<DrawRecordHit>();
    std::optional<Counter> first;
    for (const auto& record : job.records->records) {
        Counter outcome = Counter::MissFetchError;
        try {
            outcome = fetchDrawRecord(job, *record, *hit);
        } catch (const std::exception&) {
            outcome = Counter::MissFetchError;
        }
        if (outcome == Counter::Hits) {
            hit->record = record;
            return hit;
        }
        if (!first) first = outcome;
    }
    DrawRecords::Add(first.value_or(Counter::MissNoFetchPlan));
    return nullptr;
}

void Driver::commitRecordDraw(const LeanDrawJob& job, const LeanAnchor& anchor, std::shared_ptr<DrawRecordHit> hit) {
    auto writes = hit->record->writes;
    auto reads = std::move(hit->reads);
    const auto& graphics = job.shape->decode->state;
    const VulkanDevice::DrawTraceContext trace{0, job.serial, job.queue, static_cast<std::size_t>(job.packetOffset), graphics.hasColorTarget ? graphics.color.address : 0};
    auto commit = [this, hit = std::move(hit), localDevice = anchor.device, decode = job.shape->decode, queue = job.queue, opcode = job.opcode, epoch = job.epoch, trace, drawKey = job.drawKey, shape = job.shapeKey] {
        static const std::vector<std::shared_ptr<DispatchVariant>> noRecipeStages;
        DrawPipeline::FollowEpoch(epoch);
        GuestMemory::SetCurrentPacket(opcode, queue);
        const auto& record = *hit->record;
        commitDraw(localDevice, queue, decode->state, hit->parameters, hit->stages, hit->snapshots, nullptr, noRecipeStages, drawKey, trace, shape, record.contentKey.empty() ? nullptr : &record.contentKey, record.fragmentOutputs);
    };
    DrawPipeline::Queue0().Enqueue(std::move(commit), std::move(writes), 0, {}, std::move(reads), true);
}

void Driver::finishRecordDraw(const LeanDrawJob& job) {
    using DrawRecords::Counter;
    const auto& anchor = leanDraws.AnchorAt(job.anchor);
    const bool verify = (job.recordFlags & DrawRecordFlags::VerifyDue) != 0;
    std::shared_ptr<DrawRecordHit> hit;
    DrawRecordView answer;
    if (job.records != nullptr) {
        hit = fetchDrawRecords(job);
        if (hit != nullptr) {
            DrawRecords::Add(Counter::Hits);
            if (hit->inputsMoved) DrawRecords::Add(Counter::InputRefetches);
            if (!verify) {
                commitRecordDraw(job, anchor, std::move(hit));
                return;
            }
            DrawRecords::Add(Counter::HitsOnTodaysPath);
            answer.Capture(hit->stages, hit->parameters);
            answer.reads = hit->reads;
            answer.writes = hit->record->writes;
            answer.contentKey = hit->record->contentKey;
            answer.fragmentOutputs = hit->record->fragmentOutputs;
            if (hit->record->plan->addressed) answer.CaptureSnapshots(hit->snapshots);
            if (DrawRecords::CorruptingAnswers()) answer.parameters[3] ^= 1u;
        }
    }
    struct SinkStorage {};
    auto& sink = HostThreadLocal<DrawRecordSink, SinkStorage>();
    sink.Reset();
    sink.plan = hit == nullptr && (job.recordFlags & DrawRecordFlags::BuildWanted) != 0;
    sink.verify = hit != nullptr;
    struct SinkScope {
        ~SinkScope() { DrawRecordSinkSlot::Set(nullptr); }
    } scope;
    DrawRecordSinkSlot::Set(&sink);
    finishLeanDraw(job, anchor);
    DrawRecordSinkSlot::Set(nullptr);
    if (hit != nullptr) verifyRecordHit(job, *hit->record, answer, sink);
    settleDrawRecord(job, sink, hit != nullptr);
}

void Driver::verifyRecordHit(const LeanDrawJob& job, const DrawRecord& record, const DrawRecordView& answer, const DrawRecordSink& sink) {
    using DrawRecords::Field;
    DrawRecords::NoteChecked(DrawRecords::Item::Fetch);
    if (!sink.queued) {
        DrawRecords::NoteAllowed(DrawRecords::Allowed::TodayNotQueued);
        return;
    }
    const auto& today = sink.today;
    const auto at = drawText(job, record.key);
    bool differs = false;
    const auto note = [&](Field field, const std::string& detail) {
        DrawRecords::NoteDifference(field, at + ": " + detail);
        differs = true;
    };
    if (answer.stages.size() != today.stages.size()) {
        note(Field::Variants, "the record has " + std::to_string(answer.stages.size()) + " stages, today's draw " + std::to_string(today.stages.size()));
    } else {
        for (std::size_t i = 0; i < answer.stages.size(); ++i) {
            const auto& kept = answer.stages[i];
            const auto& now = today.stages[i];
            const auto stage = "stage " + std::to_string(i) + ": ";
            if (kept.variant != now.variant || kept.stage != now.stage) note(Field::Variants, stage + "the record's variant " + hex(kept.variant) + ", today's " + hex(now.variant));
            if (kept.pushOffset != now.pushOffset) note(Field::PushConstants, stage + "the record pushes at byte " + std::to_string(kept.pushOffset) + ", today's draw at " + std::to_string(now.pushOffset));
            if (kept.bindings != now.bindings) note(Field::Bindings, stage + bindingDifference(kept.bindings, now.bindings));
            if (kept.vertexAttributes != now.vertexAttributes) note(Field::VertexAttributes, stage + firstDifference(kept.vertexAttributes, now.vertexAttributes));
            if (kept.views != now.views) note(Field::ImageViews, stage + "the record holds " + std::to_string(kept.views.size()) + " registered image views, today's draw " + std::to_string(now.views.size()));
        }
    }
    if (answer.pushed != today.pushed || answer.push != today.push) {
        std::size_t dword = 0;
        while (dword * 4 < answer.push.size() && std::memcmp(answer.push.data() + dword * 4, today.push.data() + dword * 4, 4) == 0) ++dword;
        note(Field::PushConstants, "the assembled push constants differ at dword " + std::to_string(dword));
    }
    if (answer.parameters != today.parameters) {
        std::size_t index = 0;
        while (index < answer.parameters.size() && answer.parameters[index] == today.parameters[index]) ++index;
        note(Field::Parameters, "field " + std::to_string(index) + " is " + hex(answer.parameters[index]) + " in the record, " + hex(today.parameters[index]) + " today");
    }
    if (answer.reads != today.reads) note(Field::ReadRanges, "the record reads " + std::to_string(answer.reads.size()) + " ranges, today's draw " + std::to_string(today.reads.size()));
    if (answer.writes != today.writes) note(Field::WriteTemplates, "the record holds " + std::to_string(answer.writes.size()) + " write ranges, today's draw writes " + std::to_string(today.writes.size()));
    if (answer.contentKey != today.contentKey) note(Field::ContentKeys, firstDifference(answer.contentKey, today.contentKey));
    if (record.vertexLayout != sink.vertexLayout) note(Field::VertexLayouts, firstDifference(record.vertexLayout, sink.vertexLayout));
    if (record.plan->addressed && answer.snapshotWords != today.snapshotWords) note(Field::Snapshots, "the record's snapshots cover " + std::to_string(answer.snapshotWords.size()) + " words, today's " + std::to_string(today.snapshotWords.size()) + "; " + firstDifference(answer.snapshotWords, today.snapshotWords));
    if (answer.fragmentOutputs.has_value() != today.fragmentOutputs.has_value()) DrawRecords::NoteAllowed(DrawRecords::Allowed::FragmentOutputsOneSide);
    else if (answer.fragmentOutputs != today.fragmentOutputs) note(Field::FragmentOutputs, "the record's fragment outputs " + hex(*answer.fragmentOutputs) + ", today's " + hex(*today.fragmentOutputs));
    if (!differs) return;
    auto demote = std::make_shared<DrawRecordAnswer>();
    demote->key = record.key;
    demote->demote = true;
    DrawFinisher::AnswerLean(std::move(demote));
}

void Driver::settleDrawRecord(const LeanDrawJob& job, const DrawRecordSink& sink, bool hit) {
    using DrawRecords::Counter;
    if (hit || (job.recordFlags & DrawRecordFlags::BuildWanted) == 0) return;
    if (!sink.queued || sink.contentKey.empty()) {
        DrawRecords::Add(Counter::BuildsRefused);
        return;
    }
    const auto content = sink.Content();
    const auto signature = sink.built != nullptr ? sink.built->signature : 0;
    if (job.records != nullptr) {
        for (const auto& candidate : job.records->records) {
            if (candidate->content == content && candidate->generation == job.recordGeneration && candidate->deviceSerial == job.deviceSerial && (sink.built == nullptr || candidate->signature == signature)) return;
        }
    }
    if (Graphics::RuntimeSwitches::Generation() != job.recordGeneration) {
        DrawRecords::Add(Counter::BuildsRefused);
        return;
    }
    auto answer = std::make_shared<DrawRecordAnswer>();
    answer->key = DrawRecordKey::For(job.shapeKey, job.parameters);
    answer->built = DrawRecord::Build(answer->key, job.shape->shared_from_this(), job.deviceSerial, job.recordGeneration, Graphics::Recorder::Presents(), sink);
    DrawRecords::Add(Counter::RecordsBuilt);
    DrawRecords::Add(sink.built != nullptr ? Counter::PlansBuilt : Counter::PlansRefused);
    if (!DrawFinisher::AnswerLean(std::move(answer))) DrawRecords::Add(Counter::BuildsDropped);
}

void Driver::installDrawRecord(const LeanDrawJob& job, const DrawRecordAnswer& answer) {
    using DrawRecords::Counter;
    auto& lean = leanDraws;
    const auto front = DrawFinisher::Get().Front();
    if (job.recordGeneration != lean.stamp || job.deviceSerial != lean.identitiesSerial) {
        if (answer.built != nullptr) DrawRecords::AddOnWorker(Counter::BuildsDropped);
        return;
    }
    if (answer.demote) {
        if (lean.identities.Demote(answer.key, answer.key.Probe(), front)) DrawRecords::AddOnWorker(Counter::Demotions);
        return;
    }
    if (answer.built == nullptr) return;
    const auto installed = lean.identities.Install(answer.built, Graphics::Recorder::Presents(), front);
    DrawRecords::AddOnWorker(installed == DrawRecordStore::Installed::Yes ? Counter::RecordsInstalled : Counter::BuildsDropped);
}

void Driver::retireDrawRecord(const std::byte* payload, std::shared_ptr<const void>&& answer) {
    if (answer == nullptr) return;
    const auto& job = *std::launder(reinterpret_cast<const LeanDrawJob*>(payload));
    const auto reply = std::static_pointer_cast<const DrawRecordAnswer>(std::move(answer));
    Get().installDrawRecord(job, *reply);
}

}
