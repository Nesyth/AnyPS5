#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRecords.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/LeanDraw.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <new>
#include <string>

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
    return record;
}

std::size_t DrawRecord::Bytes() const {
    return sizeof(DrawRecord) + 4 * (contentKey.capacity() + vertexLayout.capacity() + vertexInputs.capacity()) + sizeof(DrawPipeline::Range) * writes.capacity();
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
            if (kept->content == record->content || kept->shape != record->shape) continue;
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
}};

constexpr std::array<const char*, AllowedKinds> AllowedNames{"fragment outputs known on one side only"};

constexpr std::array<const char*, CounterCount> CounterNames{
    "draws", "hits", "input refetches", "hits on today's path", "new identity", "second sighting", "build pending", "shape replaced", "refused",
    "content differs", "generation moved", "not queued", "keyless", "today's draw", "indirect", "lean front verify", "inline",
    "records built", "builds refused", "records installed", "builds dropped", "identities evicted", "records cleared", "record mode changes", "demotions",
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
            counted(counts, Counter::MissNewIdentity, Counter::MissKeyless).c_str(), counted(counts, Counter::OffTodaysDraw, Counter::OffInline).c_str(), counted(counts, Counter::RecordsBuilt, Counter::Demotions).c_str(),
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

void Driver::settleDrawRecord(const LeanDrawJob& job, const DrawRecordSink& sink) {
    using DrawRecords::Counter;
    const bool probed = job.records != nullptr;
    const bool build = (job.recordFlags & DrawRecordFlags::BuildWanted) != 0;
    if (!sink.queued || sink.contentKey.empty()) {
        if (probed) DrawRecords::Add(sink.queued ? Counter::MissKeyless : Counter::MissNotQueued);
        if (build) DrawRecords::Add(Counter::BuildsRefused);
        return;
    }
    const auto content = sink.Content();
    if (probed) {
        const DrawRecord* match = nullptr;
        bool moved = false;
        for (const auto& candidate : job.records->records) {
            if (candidate->content != content) continue;
            if (candidate->generation != job.recordGeneration || candidate->deviceSerial != job.deviceSerial) {
                moved = true;
                continue;
            }
            match = candidate.get();
            break;
        }
        if (match != nullptr) {
            DrawRecords::Add(Counter::Hits);
            if (match->inputs != sink.Inputs()) DrawRecords::Add(Counter::InputRefetches);
            DrawRecords::Add(Counter::HitsOnTodaysPath);
            if ((job.recordFlags & DrawRecordFlags::VerifyDue) == 0) return;
            DrawRecords::NoteChecked(DrawRecords::Item::Fetch);
            const auto at = "draw at submission " + std::to_string(job.serial) + " DWORD " + std::to_string(job.packetOffset) + " (" + keyText(match->key) + ")";
            bool differs = false;
            if (match->contentKey != sink.contentKey) {
                DrawRecords::NoteDifference(DrawRecords::Field::ContentKeys, at + ": the record's content key of " + std::to_string(match->contentKey.size()) + " words hashes like today's key of " + std::to_string(sink.contentKey.size()) + " words but differs");
                differs = true;
            }
            if (match->vertexLayout != sink.vertexLayout) {
                DrawRecords::NoteDifference(DrawRecords::Field::VertexLayouts, at + ": the record's vertex layout of " + std::to_string(match->vertexLayout.size()) + " words hashes like today's of " + std::to_string(sink.vertexLayout.size()) + " words but differs");
                differs = true;
            }
            if (match->writes != sink.writes) {
                DrawRecords::NoteDifference(DrawRecords::Field::WriteTemplates, at + ": the record holds " + std::to_string(match->writes.size()) + " write ranges, today's draw writes " + std::to_string(sink.writes.size()));
                differs = true;
            }
            if (match->fragmentOutputs.has_value() != sink.fragmentOutputs.has_value()) {
                DrawRecords::NoteAllowed(DrawRecords::Allowed::FragmentOutputsOneSide);
            } else if (match->fragmentOutputs != sink.fragmentOutputs) {
                DrawRecords::NoteDifference(DrawRecords::Field::FragmentOutputs, at + ": the record's fragment outputs " + hex(*match->fragmentOutputs) + ", today's " + hex(*sink.fragmentOutputs));
                differs = true;
            }
            if (sink.shape != match->key.shape) {
                DrawRecords::NoteDifference(DrawRecords::Field::Keys, at + ": today's draw keyed shape " + hex(sink.shape) + ", the record " + hex(match->key.shape));
                differs = true;
            }
            if (!differs) return;
            auto answer = std::make_shared<DrawRecordAnswer>();
            answer->key = match->key;
            answer->demote = true;
            DrawFinisher::AnswerLean(std::move(answer));
            return;
        }
        DrawRecords::Add(moved ? Counter::MissGenerationMoved : Counter::MissContentDiffers);
    }
    if (!build) return;
    if (Graphics::RuntimeSwitches::Generation() != job.recordGeneration) {
        DrawRecords::Add(Counter::BuildsRefused);
        return;
    }
    auto answer = std::make_shared<DrawRecordAnswer>();
    answer->key = DrawRecordKey::For(job.shapeKey, job.parameters);
    answer->built = DrawRecord::Build(answer->key, job.shape->shared_from_this(), job.deviceSerial, job.recordGeneration, Graphics::Recorder::Presents(), sink);
    DrawRecords::Add(Counter::RecordsBuilt);
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
