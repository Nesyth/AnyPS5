#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRecords.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DirectDraws.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawTickets.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/LeanDraw.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/StagePlanReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadReport.hpp"
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
#include <optional>
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
    refusal = DrawRecordPlanRefusal::None;
    built = nullptr;
    shape = 0;
    contentKey.clear();
    vertexLayout.clear();
    vertexInputs.clear();
    fragmentOutputs.reset();
    writes.clear();
}

void DrawVertexLayout(std::span<const std::optional<ShaderRecompiler::ShaderVertexStageInfo>> infos, std::vector<std::uint32_t>& layout, std::vector<std::uint32_t>& inputs) {
    layout.clear();
    inputs.clear();
    for (const auto& info : infos) {
        if (!info) {
            layout.push_back(0);
            continue;
        }
        const auto count = std::min<std::uint32_t>(info->resourcesNum, ShaderRecompiler::ShaderVertexStageInfo::MaxResources);
        layout.insert(layout.end(), {0x80000000u | count, info->fetchAttribReg, info->fetchBufferReg, info->fetchEmbedded ? 1u : 0u, info->paClVsOutCntl, info->depthRemapMode, info->pointList ? 1u : 0u, std::bit_cast<std::uint32_t>(info->pointSize), std::bit_cast<std::uint32_t>(info->pointSizeMin), std::bit_cast<std::uint32_t>(info->pointSizeMax)});
        for (std::uint32_t index = 0; index < count; ++index) {
            const auto& fields = info->resources[index].fields;
            const auto& destination = info->resourcesDst[index];
            layout.insert(layout.end(), {fields[1] & 0xffff0000u, fields[3], static_cast<std::uint32_t>(destination.registerStart), static_cast<std::uint32_t>(destination.registersNum), static_cast<std::uint32_t>(destination.attrId), destination.fetchIndex});
            inputs.insert(inputs.end(), {fields[0], fields[1] & 0xffffu, fields[2]});
        }
    }
}

void DrawRecordSink::NoteVertices(std::span<const std::optional<ShaderRecompiler::ShaderVertexStageInfo>> infos) {
    DrawVertexLayout(infos, vertexLayout, vertexInputs);
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
    return sizeof(DrawRecord) + 4 * (contentKey.capacity() + vertexLayout.capacity() + vertexInputs.capacity()) + sizeof(DrawPipeline::Range) * writes.capacity() + sizeof(DrawRecordProbe) * probes.capacity() + (plan != nullptr ? plan->bytes : 0);
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

namespace {

struct ProbeWord {
    std::int16_t base = -1;
    std::uint64_t address = 0;
    std::uint32_t value = 0;

    bool Before(const ProbeWord& other) const { return base < other.base || (base == other.base && address < other.address); }
};

void StageProbeWords(const DrawRecordStage& stage, std::vector<ProbeWord>& out) {
    out.clear();
    if (stage.variant == nullptr) return;
    const auto& variant = *stage.variant;
    std::size_t offset = 0;
    std::size_t data = 0;
    for (std::size_t run = 0; run < variant.runs.size(); ++run) {
        const auto [begin, end] = variant.runs[run];
        const auto count = static_cast<std::size_t>((end - begin) / sizeof(std::uint32_t));
        const bool based = run < stage.runBases.size() && stage.runBases[run] >= 0 && static_cast<std::size_t>(stage.runBases[run]) < stage.userData.size();
        const auto base = based ? static_cast<std::int16_t>(stage.runBases[run]) : std::int16_t{-1};
        const auto origin = based ? static_cast<std::uint64_t>(stage.userData[static_cast<std::size_t>(base)] & ~std::uint32_t{3}) : std::uint64_t{0};
        for (std::size_t word = 0; word < count && offset + word < variant.words.size(); ++word) {
            const auto position = offset + word;
            while (data < variant.dataPositions.size() && variant.dataPositions[data] < position) ++data;
            if (data < variant.dataPositions.size() && variant.dataPositions[data] == position) continue;
            out.push_back({base, begin + word * sizeof(std::uint32_t) - origin, variant.words[position]});
        }
        offset += count;
    }
    std::sort(out.begin(), out.end(), [](const ProbeWord& a, const ProbeWord& b) { return a.Before(b); });
}

}

std::vector<DrawRecordProbe> DrawRecordProbes(const DrawRecordPlan& plan, const DrawRecordSet* kept) {
    std::vector<DrawRecordProbe> probes;
    if (kept == nullptr || kept->records.empty()) return probes;
    const auto stages = plan.stages.size();
    std::vector<std::vector<ProbeWord>> mine(stages);
    std::vector<std::vector<ProbeWord>> theirs(stages);
    for (std::size_t i = 0; i < stages; ++i) StageProbeWords(plan.stages[i], mine[i]);
    const auto told = [&](const DrawRecordProbe& probe) {
        const auto& words = theirs[probe.stage];
        const ProbeWord key{probe.base, probe.address, 0};
        const auto found = std::lower_bound(words.begin(), words.end(), key, [](const ProbeWord& a, const ProbeWord& b) { return a.Before(b); });
        return found != words.end() && !key.Before(*found) && found->value != probe.value;
    };
    for (const auto& record : kept->records) {
        if (probes.size() >= DrawRecordStore::MaxProbes) break;
        if (record->plan == nullptr || record->plan->stages.size() != stages) continue;
        for (std::size_t i = 0; i < stages; ++i) StageProbeWords(record->plan->stages[i], theirs[i]);
        if (std::any_of(probes.begin(), probes.end(), told)) continue;
        std::size_t added = 0;
        for (std::size_t i = 0; i < stages && added < DrawRecordStore::ProbesPerVariant && probes.size() < DrawRecordStore::MaxProbes; ++i) {
            const auto& ours = mine[i];
            const auto& other = theirs[i];
            for (std::size_t a = 0, b = 0; a < ours.size() && b < other.size() && added < DrawRecordStore::ProbesPerVariant && probes.size() < DrawRecordStore::MaxProbes;) {
                if (ours[a].Before(other[b])) {
                    ++a;
                    continue;
                }
                if (other[b].Before(ours[a])) {
                    ++b;
                    continue;
                }
                if (other[b].value != ours[a].value) {
                    probes.push_back({ours[a].address, ours[a].value, static_cast<std::uint16_t>(i), ours[a].base});
                    ++added;
                }
                ++a;
                ++b;
            }
        }
    }
    return probes;
}

std::optional<std::size_t> DrawRecordTwin(const DrawRecordSet& set, const DrawRecordHash& content, std::uint64_t generation, std::uint64_t deviceSerial, const std::uint64_t* signature) {
    for (std::size_t index = 0; index < set.records.size(); ++index) {
        const auto& record = *set.records[index];
        if (record.content == content && record.generation == generation && record.deviceSerial == deviceSerial && (signature == nullptr || record.signature == *signature)) return index;
    }
    return std::nullopt;
}

bool DrawRecordProbesHold(std::span<const DrawRecordProbe> probes, std::span<const std::vector<std::uint32_t>> users) {
    for (const auto& probe : probes) {
        auto address = probe.address;
        if (probe.base >= 0) {
            if (probe.stage >= users.size() || static_cast<std::size_t>(probe.base) >= users[probe.stage].size()) return false;
            address += users[probe.stage][static_cast<std::size_t>(probe.base)] & ~std::uint32_t{3};
        }
        std::uint32_t word = 0;
        if (GuestMemory::CopyMapped(address, std::as_writable_bytes(std::span(&word, 1))) != GuestMemory::Compare::Equal || word != probe.value) return false;
    }
    return true;
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

std::size_t DrawRecordStore::ConfiguredByteCap() {
    static const std::size_t cap = [] {
        const char* text = std::getenv("APS5_DRAW_RECORD_MIB");
        if (text == nullptr || text[0] == '\0') return DefaultByteCap;
        char* end = nullptr;
        const auto mib = std::strtoull(text, &end, 10);
        if (end == text || *end != '\0' || mib == 0 || mib > 4096) return DefaultByteCap;
        return static_cast<std::size_t>(mib) << 20;
    }();
    return cap;
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
    {Item::Fetch, "fetched runs"},
    {Item::Commit, "templates"},
    {Item::Commit, "pipelines"},
    {Item::Commit, "framebuffers"},
    {Item::Commit, "target views"},
    {Item::Commit, "targets"},
    {Item::Commit, "pass keys"},
    {Item::Commit, "moved buffers"},
    {Item::Commit, "push constants"},
    {Item::Commit, "vertex inputs"},
    {Item::Commit, "shader stages"},
    {Item::Commit, "fragment outputs"},
    {Item::Commit, "bound buffers"},
    {Item::Commit, "pass access"},
    {Item::Commit, "batch keeps"},
    {Item::Commit, "target notes"},
}};

constexpr std::array<const char*, AllowedKinds> AllowedNames{"fragment outputs known on one side only", "today's draw not queued", "the record's commit fell back", "today's commit built no recipe", "today's commit served another template"};

constexpr std::array<const char*, CounterCount> CounterNames{
    "draws", "hits", "input refetches", "hits on today's path", "relocated base and value words", "moved buffer bases patched", "new identity", "second sighting", "build pending", "shape replaced", "refused",
    "content differs", "generation moved", "not queued", "keyless", "no fetch plan", "user word moved", "hidden user word moved", "relocated user word misaligned", "value user word zero", "walked user word moved", "callee user word moved", "base and value user word moved", "user words unreadable", "fetch gate", "image views moved", "vertex layout differs", "moved buffer fields", "memory changed", "fetch error",
    "today's draw", "indirect", "lean front verify", "inline",
    "records built", "records rebuilt past the candidates", "fetch plans built", "fetch plans refused", "builds refused", "records installed", "builds dropped", "dropped without an answer slot", "dropped as stale", "dropped over the byte cap", "dropped without an identity", "dropped for a refused identity", "identities evicted", "records cleared", "record mode changes", "demotions",
    "recorded", "pushed", "moved without push", "targets held by the pass", "kept once per batch", "target notes skipped", "moved buffers trusted", "committer halves built", "verified on today's path",
    "no committer half", "not recordable", "targets", "template gone", "objects gone", "texture proof", "moved buffers", "no recipe built", "demoted",
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
std::atomic<bool> corruptingCommits{false};

}

void CorruptAnswers(bool corrupt) {
    corrupting.store(corrupt, std::memory_order_relaxed);
}

bool CorruptingAnswers() {
    return corrupting.load(std::memory_order_relaxed);
}

void CorruptCommits(bool corrupt) {
    corruptingCommits.store(corrupt, std::memory_order_relaxed);
}

bool CorruptingCommits() {
    return corruptingCommits.load(std::memory_order_relaxed);
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

void AddOnCommitter(Counter counter, std::uint64_t count) {
    AddOnWorker(counter, count);
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
    std::uint64_t commits = 0;
    for (auto index = static_cast<std::size_t>(Counter::CommitHits); index <= static_cast<std::size_t>(Counter::CommitDemoted); ++index) commits += counts[index];
    if (draws != 0 || commits != 0 || at(counts, Counter::RecordModeChanges) != 0 || at(counts, Counter::RecordsCleared) != 0) {
        const auto hits = at(counts, Counter::Hits);
        AgcDriver::ProfilePrint_nid_no_patch("[draw-records] %.1f s: draws %llu, hits %llu (%.1f%%), input refetches %llu, hits on today's path %llu, relocated base and value words %llu, moved buffer bases patched %llu; misses: %s; records off: %s; store: %s; %llu identities, %llu records, %.1f KiB; commit: %s; commit misses: %s\n", seconds,
            static_cast<unsigned long long>(draws), static_cast<unsigned long long>(hits), draws != 0 ? 100.0 * static_cast<double>(hits) / static_cast<double>(draws) : 0.0,
            static_cast<unsigned long long>(at(counts, Counter::InputRefetches)), static_cast<unsigned long long>(at(counts, Counter::HitsOnTodaysPath)),
            static_cast<unsigned long long>(at(counts, Counter::HitsBaseValue)), static_cast<unsigned long long>(at(counts, Counter::HitsBuffersPatched)),
            counted(counts, Counter::MissNewIdentity, Counter::MissFetchError).c_str(), counted(counts, Counter::OffTodaysDraw, Counter::OffInline).c_str(), counted(counts, Counter::RecordsBuilt, Counter::Demotions).c_str(),
            static_cast<unsigned long long>(storeIdentities.load(std::memory_order_relaxed)), static_cast<unsigned long long>(storeRecords.load(std::memory_order_relaxed)), static_cast<double>(storeBytes.load(std::memory_order_relaxed)) / 1024.0,
            counted(counts, Counter::CommitHits, Counter::CommitVerified).c_str(), counted(counts, Counter::CommitFirst, Counter::CommitDemoted).c_str());
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

namespace {

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

struct FetchScratchStorage {};

struct FetchScratch {
    std::vector<std::vector<std::uint32_t>> userData;
    std::vector<std::vector<std::uint32_t>> liveWords;
    std::vector<std::vector<Graphics::DecodeRead>> decodeReads;
    std::vector<ShaderRecompiler::MemoryRegion> regions;
    std::vector<ShaderRecompiler::MemoryRegion> memory;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> live;
    DispatchVariant moved;
};

DrawRecords::Counter PrecheckDrawRecord(const LeanDrawJob& job, const DrawRecord& record, std::span<const std::vector<std::uint32_t>> users) {
    using DrawRecords::Counter;
    const auto* plan = record.plan.get();
    if (plan == nullptr) return Counter::MissNoFetchPlan;
    if (record.generation != job.recordGeneration || record.deviceSerial != job.deviceSerial || record.shape.get() != job.shape) return Counter::MissGenerationMoved;
    if (plan->stages.size() != users.size()) return Counter::MissNoFetchPlan;
    for (std::size_t i = 0; i < users.size(); ++i) {
        const auto& stage = plan->stages[i];
        const auto& user = users[i];
        if (user.size() != stage.userData.size()) return Counter::MissUserWordUnread;
        for (std::size_t index = 0; index < user.size(); ++index) {
            const auto from = stage.userData[index];
            const auto to = user[index];
            if (from == to) continue;
            if (index < stage.hiddenUserWords || index >= 64u) return Counter::MissUserWordHidden;
            const auto bit = std::uint64_t{1} << index;
            if ((stage.relocatableWords & bit) != 0 || (stage.baseValueWords & bit) != 0) {
                if (from == 0 || to == 0 || ((from ^ to) & 3u) != 0) return Counter::MissUserWordRelocation;
            } else if ((stage.valueWords & bit) != 0) {
                if (from == 0 || to == 0) return Counter::MissUserWordValue;
            } else if ((stage.walkedWords & bit) != 0) {
                if ((stage.calleeWords & bit) != 0) return Counter::MissUserWordCallee;
                if ((stage.valueUseWords & bit) != 0) return Counter::MissUserWordBaseValue;
                return Counter::MissUserWordWalked;
            } else if ((stage.vertexWords & bit) == 0 && (stage.forwardedWords & bit) == 0 && (stage.userDataMask & bit) != 0) {
                return Counter::MissUserWord;
            }
        }
    }
    return DrawRecordProbesHold(record.probes, users) ? Counter::Hits : Counter::MissContentDiffers;
}

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
    if (const auto& indirect = drawParameters.indirect) {
        using Rule = Pm4::DrawParameters::IndirectDraw::Rule;
        parameters[6] = indirect->arguments | static_cast<std::uint64_t>(indirect->vertexRule == Rule::InPlace ? 1u : 0u) << 56u | static_cast<std::uint64_t>(indirect->instanceRule == Rule::InPlace ? 1u : 0u) << 57u | static_cast<std::uint64_t>(indirect->countIndirect ? 1u : 0u) << 58u;
        parameters[7] = indirect->vertexConstant | static_cast<std::uint64_t>(indirect->instanceConstant) << 32u;
    }
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

std::shared_ptr<const DrawRecordPlan> Driver::buildDrawRecordPlan(const Graphics::State& graphics, const Pm4::DrawParameters& parameters, const std::vector<DrawProgram>& programs, const std::vector<ShaderRecompiler::ProgramRole>& roles, const std::vector<StageCapture>& stageCaptures, const std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>& vertexInfos, const std::vector<std::vector<Graphics::DecodeRead>>& decodeReads, std::span<const Graphics::CompiledShader> stages, DrawRecordPlanRefusal* refusal) {
    using Role = ShaderRecompiler::ProgramRole;
    using DescriptorRole = ShaderRecompiler::DescriptorRole;
    const auto refuse = [&](DrawRecordPlanRefusal reason) -> std::shared_ptr<const DrawRecordPlan> {
        if (refusal != nullptr) *refusal = reason;
        return nullptr;
    };
    if (parameters.indirect) return refuse(DrawRecordPlanRefusal::Indirect);
    if (graphics.rectList || graphics.stages.mesh || graphics.stages.tessellation || graphics.stages.tessellationGeometry || graphics.stages.path != Graphics::ShaderPath::Vertex) return refuse(DrawRecordPlanRefusal::Shape);
    const auto count = programs.size();
    if (count == 0 || stages.size() != count || stageCaptures.size() != count || roles.size() != count || vertexInfos.size() != count || decodeReads.size() != count) return refuse(DrawRecordPlanRefusal::Stages);
    auto plan = std::make_shared<DrawRecordPlan>();
    plan->stages.resize(count);
    std::uint64_t signature = 0xcbf29ce484222325ull;
    const auto sign = [&](std::uint64_t value) { signature = mixWord(signature, value); };
    for (std::size_t i = 0; i < count; ++i) {
        const auto& program = programs[i];
        const auto& capture = stageCaptures[i];
        if (roles[i] == Role::GeometryBack) return refuse(DrawRecordPlanRefusal::GeometryBack);
        if (capture.compiled == nullptr || capture.capture == nullptr || stages[i].program != capture.compiled.get() || stages[i].heap != nullptr) return refuse(DrawRecordPlanRefusal::Capture);
        if (capture.direct || capture.ring || capture.heapWords != nullptr) return refuse(DrawRecordPlanRefusal::Direct);
        if (program.snapshot == nullptr || program.snapshot->header.empty() || program.tailContinuation != nullptr) return refuse(DrawRecordPlanRefusal::Program);
        const auto& compiled = *capture.compiled;
        if (compiled.gpuResolvedImages || compiled.heapTemplateId != 0 || compiled.deviceImageTables || compiled.dataAddressPush || !CacheableResult(compiled)) return refuse(DrawRecordPlanRefusal::Result);
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
        stage.valueUseWords = capture.userDataUse.asValue;
        auto variant = std::make_shared<DispatchVariant>();
        variant->compiled = capture.compiled;
        variant->registeredViews = capture.registeredViews;
        variant->shader = program.snapshot;
        variant->forgetSerial = capture.forgetSerial;
        variant->pushOffset = capture.pushOffset;
        variant->userDataMask = capture.userDataMask;
        std::vector<ShaderRecompiler::MemoryRegion> regions(capture.regions.begin(), capture.regions.end());
        std::stable_sort(regions.begin(), regions.end(), [](const ShaderRecompiler::MemoryRegion& a, const ShaderRecompiler::MemoryRegion& b) { return a.guestAddress < b.guestAddress; });
        const auto users = std::min<std::size_t>(program.userData.size(), 64);
        const auto below = [](std::size_t bits) { return bits >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << bits) - 1; };
        const auto& walked = capture.userDataUse;
        const auto both = walked.asBase & walked.asValue & below(users) & ~below(program.hiddenUserWords);
        if (both == 0) {
            tagVariantRuns(*variant, regions, walked, true, program.hiddenUserWords, capture.supplied);
        } else {
            auto use = walked;
            use.asValue &= ~both;
            tagVariantRuns(*variant, regions, use, true, program.hiddenUserWords, capture.supplied);
        }
        tagUserValueWords(*variant, program, nullptr);
        auto baseValue = variant->relocatableWords & both;
        variant->relocatableWords &= ~both;
        if (!capture.capture->readTrace.leaves.empty()) recordStageDataPositions(*variant, *capture.capture, decodeReads[i]);
        stage.runBases = variant->runBases;
        stage.relocatableWords = variant->relocatableWords;
        stage.valueWords = variant->valueWords;
        stage.valueSlots = variant->valueSlots;
        stage.walkedWords |= variant->calleeWords;
        stage.calleeWords = variant->calleeWords;
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
        const auto userWord = [&](std::uint32_t value) { return std::find(program.userData.begin(), program.userData.end(), value) != program.userData.end(); };
        const auto* flatSlots = flat ? &compiled.bindings[flatBinding].guestDescriptor : nullptr;
        bool explained = true;
        for (std::size_t slot = 0; flat && explained && slot < flatSlots->size(); ++slot) {
            const bool leaf = std::any_of(targets.begin(), targets.end(), [&](const DrawRecordTarget& target) { return target.binding == flatBinding && target.word == slot; });
            if (!leaf && countIn(words, (*flatSlots)[slot]) == 0 && !userWord((*flatSlots)[slot])) explained = false;
        }
        bool derived = false;
        for (const auto& binding : compiled.bindings) {
            if (binding.role != DescriptorRole::GuestBuffers) continue;
            if (binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 4u) {
                derived = true;
                break;
            }
            for (std::uint32_t element = 0; element < binding.count && !derived; ++element) {
                const auto* descriptor = binding.guestDescriptor.data() + element * 4u;
                derived = BufferPresent(descriptor) && countIn(words, descriptor[0]) == 0 && !userWord(descriptor[0]);
            }
        }
        std::vector<std::uint32_t> pushWords(compiled.pushConstants.size() / sizeof(std::uint32_t));
        if (!pushWords.empty()) std::memcpy(pushWords.data(), compiled.pushConstants.data(), pushWords.size() * sizeof(std::uint32_t));
        const auto& forwarded = compiled.shaderDataUserWords;
        const auto forwardedFrom = [&](std::size_t slot, std::uint32_t index) { return slot < forwarded.size() && forwarded[slot] == index; };
        for (std::uint32_t index = 0; index < users; ++index) {
            const auto bit = std::uint64_t{1} << index;
            if ((baseValue & bit) == 0) continue;
            const auto value = program.userData[index];
            bool refused = derived || !explained || value == 0 || index + 2u >= program.userData.size() || countIn(words, value) != 0 || countIn(program.userData, value) != 1 || std::any_of(compiled.specialization.begin(), compiled.specialization.end(), [&](const auto& constant) { return constant.value == value; });
            for (std::size_t slot = 0; !refused && slot < pushWords.size(); ++slot) refused = pushWords[slot] == value && !forwardedFrom(slot, index);
            std::vector<std::array<std::uint32_t, 3>> slots;
            for (std::size_t b = 0; !refused && b < compiled.bindings.size(); ++b) {
                const auto& binding = compiled.bindings[b];
                const auto& descriptor = binding.guestDescriptor;
                for (std::size_t word = 0; !refused && word < descriptor.size(); ++word) {
                    if (descriptor[word] != value || (binding.role == DescriptorRole::ShaderData && forwardedFrom(word, index))) continue;
                    const auto element = word / 4u;
                    const bool readOnly = binding.role == DescriptorRole::GuestBuffers && descriptor.size() == static_cast<std::size_t>(binding.count) * 4u && word % 4u == 0 && element < binding.bufferWritten.size() && !binding.bufferWritten[element] && (element >= binding.bufferAtomic.size() || !binding.bufferAtomic[element]);
                    if (!readOnly || descriptor[word + 1] != program.userData[index + 1] || descriptor[word + 2] != program.userData[index + 2]) {
                        refused = true;
                        break;
                    }
                    const auto base = (static_cast<std::uint64_t>(descriptor[word + 1] & 0xffffu) << 32u) | AlignedWord(value);
                    const auto stride = (descriptor[word + 1] >> 16u) & 0x3fffu;
                    const auto bytes = std::max<std::uint64_t>(stride != 0 ? static_cast<std::uint64_t>(stride) * descriptor[word + 2] : descriptor[word + 2], sizeof(std::uint32_t));
                    refused = std::any_of(capture.userDataUse.reads.begin(), capture.userDataUse.reads.end(), [&](const auto& read) { return read.second != static_cast<std::int32_t>(index) && read.first >= base && read.first < base + bytes; });
                    slots.push_back({index, static_cast<std::uint32_t>(b), static_cast<std::uint32_t>(word)});
                }
            }
            if (refused || slots.empty()) {
                baseValue &= ~bit;
                continue;
            }
            stage.baseValueSlots.insert(stage.baseValueSlots.end(), slots.begin(), slots.end());
        }
        stage.baseValueWords = baseValue;
        const auto foreignValue = [&](std::uint32_t value) {
            for (std::size_t other = 0; other < compiled.bindings.size(); ++other) {
                const auto& kept = compiled.bindings[other];
                if (other != flatBinding && kept.role != DescriptorRole::GuestBuffers && countIn(kept.guestDescriptor, value) != 0) return true;
            }
            return false;
        };
        const auto userSlot = [&](std::size_t b, std::uint32_t element) {
            const auto names = [&](const std::array<std::uint32_t, 3>& slot) { return slot[1] == b && slot[2] / 4u == element; };
            return std::any_of(stage.valueSlots.begin(), stage.valueSlots.end(), names) || std::any_of(stage.baseValueSlots.begin(), stage.baseValueSlots.end(), names);
        };
        std::vector<DrawRecordTarget> bufferTargets;
        for (std::size_t b = 0; explained && !derived && b < compiled.bindings.size(); ++b) {
            const auto& binding = compiled.bindings[b];
            if (binding.role != DescriptorRole::GuestBuffers || binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 4u) continue;
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                if (element >= binding.bufferWritten.size() || binding.bufferWritten[element] || (element < binding.bufferAtomic.size() && binding.bufferAtomic[element])) continue;
                const auto word = element * 4u;
                if (userSlot(b, element)) continue;
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
                const auto movable = [&](std::uint32_t value) { return countIn(words, value) == 1 && countIn(pushWords, value) == 0 && !userWord(value) && !foreignValue(value); };
                if (!movable(descriptor[0])) continue;
                stage.buffers.push_back({static_cast<std::uint32_t>(b), element, *found});
                const auto strayCopy = [&](std::uint32_t field) {
                    for (std::size_t slot = 0; flat && slot < flatSlots->size(); ++slot) {
                        if ((*flatSlots)[slot] != descriptor[field]) continue;
                        if (slot < field || slot - field + 4u > flatSlots->size() || !std::equal(descriptor, descriptor + 4, flatSlots->begin() + static_cast<std::ptrdiff_t>(slot - field))) return true;
                    }
                    return false;
                };
                for (const std::uint32_t field : {0u, 2u}) {
                    const auto value = descriptor[field];
                    if (field != 0 && (flatTarget(*found + field) || !movable(value) || strayCopy(field))) continue;
                    bufferTargets.push_back({*found + field, static_cast<std::uint32_t>(b), word + field});
                    for (std::size_t slot = 0; flat && slot < flatSlots->size(); ++slot) {
                        if ((*flatSlots)[slot] == value) bufferTargets.push_back({*found + field, flatBinding, static_cast<std::uint32_t>(slot)});
                    }
                }
            }
        }
        const auto dropValue = [&](std::uint32_t value) {
            std::vector<std::uint32_t> bases;
            for (const auto& target : bufferTargets) {
                if (words[target.position] == value && target.word % 4u == 0 && compiled.bindings[target.binding].role == DescriptorRole::GuestBuffers) bases.push_back(target.position);
            }
            const auto inBase = [&](std::uint32_t position) { return std::any_of(bases.begin(), bases.end(), [&](std::uint32_t base) { return position >= base && position < base + 4u; }); };
            std::erase_if(bufferTargets, [&](const DrawRecordTarget& target) { return words[target.position] == value || inBase(target.position); });
            std::erase_if(stage.buffers, [&](const DrawRecordBuffer& buffer) { return std::find(bases.begin(), bases.end(), buffer.position) != bases.end(); });
        };
        for (bool pruned = true; pruned;) {
            pruned = false;
            for (std::size_t b = 0; b < compiled.bindings.size() && !pruned; ++b) {
                const auto& kept = compiled.bindings[b];
                if (kept.role != DescriptorRole::GuestBuffers) continue;
                for (std::size_t word = 0; word < kept.guestDescriptor.size() && !pruned; ++word) {
                    const auto value = kept.guestDescriptor[word];
                    const bool listed = std::any_of(bufferTargets.begin(), bufferTargets.end(), [&](const DrawRecordTarget& target) { return target.binding == b && target.word == word; });
                    if (listed || std::none_of(bufferTargets.begin(), bufferTargets.end(), [&](const DrawRecordTarget& target) { return words[target.position] == value; })) continue;
                    dropValue(value);
                    pruned = true;
                }
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
        const auto structural = stage.walkedWords & ~stage.relocatableWords & ~stage.valueWords & ~stage.baseValueWords;
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
        plan->bytes += sizeof(DrawRecordStage) + sizeof(DispatchVariant) + sizeof(std::uint32_t) * (stage.variant->words.size() + stage.variant->dataPositions.size() + stage.userData.size()) + sizeof(std::pair<std::uint64_t, std::uint64_t>) * stage.variant->runs.size() + sizeof(ShaderRecompiler::MemoryRegion) * stage.regions.size() + sizeof(DrawRecordTarget) * targets.size() + sizeof(std::array<std::uint32_t, 3>) * (stage.valueSlots.size() + stage.baseValueSlots.size()) + sizeof(ShaderRecompiler::RecompileResult) + compiled.pushConstants.size();
        for (const auto& binding : compiled.bindings) plan->bytes += sizeof(binding) + sizeof(std::uint32_t) * binding.guestDescriptor.size();
    }
    plan->signature = signature == 0 ? 1 : signature;
    return plan;
}

DrawRecords::Counter Driver::fetchDrawRecord(const LeanDrawJob& job, const DrawRecord& record, DrawRecordHit& hit) {
    using DrawRecords::Counter;
    const auto* plan = record.plan.get();
    if (plan == nullptr) return Counter::MissNoFetchPlan;
    if (record.generation != job.recordGeneration || record.deviceSerial != job.deviceSerial || record.shape.get() != job.shape) return Counter::MissGenerationMoved;
    const auto count = job.shape->decode->programs.size();
    auto& scratch = HostThreadLocal<FetchScratch, FetchScratchStorage>();
    if (scratch.userData.size() < count) return Counter::MissNoFetchPlan;
    return fetchRecordPlan(job, *plan, std::span<const std::vector<std::uint32_t>>(scratch.userData.data(), count), hit);
}

DrawRecords::Counter Driver::fetchRecordPlan(const LeanDrawJob& job, const DrawRecordPlan& recordPlan, std::span<const std::vector<std::uint32_t>> users, DrawRecordHit& hit, EntryOutcome* gate) {
    using DrawRecords::Counter;
    using DescriptorRole = ShaderRecompiler::DescriptorRole;
    const auto* plan = &recordPlan;
    const auto& decode = *job.shape->decode;
    const auto& programs = decode.programs;
    const auto count = programs.size();
    auto& scratch = HostThreadLocal<FetchScratch, FetchScratchStorage>();
    if (plan->stages.size() != count || users.size() < count) return Counter::MissNoFetchPlan;
    if (scratch.liveWords.size() < count) scratch.liveWords.resize(count);
    if (scratch.decodeReads.size() < count) scratch.decodeReads.resize(count);
    const bool verifying = (job.recordFlags & DrawRecordFlags::VerifyDue) != 0;
    scratch.memory.clear();
    hit.sides.assign(count, {});
    hit.words.assign(plan->addressed ? count : 0, {});
    hit.decodeReads.assign(plan->addressed ? count : 0, {});
    hit.snapshots.clear();
    hit.fetched.clear();
    hit.stages.clear();
    hit.stages.reserve(count);
    hit.inputsMoved = false;
    hit.baseValueMoved = false;
    hit.buffersPatched = false;
    hit.relocated = false;
    std::uint64_t imagesFlushed = 0, runsSynced = 0;
    std::optional<SampledReadScope> sampling;
    const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& stage = plan->stages[i];
        const auto& program = programs[i];
        const auto& compiled = *stage.compiled;
        const auto& user = users[i];
        if (user.size() != stage.userData.size()) return Counter::MissUserWordUnread;
        if (stage.registeredViews != nullptr && !CaptureRegisteredViewsCurrent(*stage.registeredViews)) return Counter::MissImageViews;
        const auto userMoved = [&](std::size_t index) { return index < user.size() && user[index] != stage.userData[index]; };
        bool relocated = false;
        for (const auto base : stage.runBases) relocated = relocated || (base >= 0 && userMoved(static_cast<std::size_t>(base)));
        for (std::size_t index = 0; index < user.size() && index < 64u; ++index) {
            if ((stage.baseValueWords >> index & 1u) != 0 && userMoved(index)) hit.baseValueMoved = true;
        }
        const DispatchVariant* variant = stage.variant.get();
        const auto* regions = &stage.regions;
        if (relocated) {
            hit.relocated = true;
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
        if (outcome != EntryOutcome::Equal && outcome != EntryOutcome::EqualData) {
            if (gate != nullptr) *gate = outcome;
            return Counter::MissFetchGate;
        }
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
                if (compiled.bindings[stage.targets[at].binding].role == DescriptorRole::GuestBuffers) hit.buffersPatched = true;
            }
            liveWords[position] = value;
        }
        for (const auto& slots : {&stage.valueSlots, &stage.baseValueSlots}) {
            for (const auto& [index, binding, word] : *slots) {
                if (userMoved(index) && !patch(binding, word, user[index])) return Counter::MissFetchError;
            }
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
        if (!verifying) continue;
        for (const auto& [begin, end] : variant->runs) {
            for (auto address = begin; address < end; address += sizeof(std::uint32_t)) hit.fetched.push_back(address);
        }
    }
    if (verifying) {
        std::sort(hit.fetched.begin(), hit.fetched.end());
        hit.fetched.erase(std::unique(hit.fetched.begin(), hit.fetched.end()), hit.fetched.end());
    }
    AppendProgramMemory(scratch.memory, programs);
    hit.parameters = job.parameters;
    const auto& front = *plan->stages.front().compiled;
    const auto& firstUser = users.front();
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

std::shared_ptr<DrawRecordHit> Driver::fetchDrawRecords(const LeanDrawJob& job, std::size_t& checked) {
    using DrawRecords::Counter;
    DrawPipeline::FollowEpoch(job.epoch);
    DrawPipeline::Active() = true;
    GuestMemory::SetCurrentPacket(job.opcode, job.queue);
    const DrawFinisher::AheadScope ahead;
    const auto& records = job.records->records;
    checked = records.size();
    const auto& programs = job.shape->decode->programs;
    const auto count = programs.size();
    auto& scratch = HostThreadLocal<FetchScratch, FetchScratchStorage>();
    if (scratch.userData.size() < count) {
        scratch.userData.resize(count);
        scratch.liveWords.resize(count);
        scratch.decodeReads.resize(count);
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (ReadRecordUserWords(programs[i], job.words, scratch.userData[i])) continue;
        DrawRecords::Add(Counter::MissUserWordUnread);
        return nullptr;
    }
    const std::span<const std::vector<std::uint32_t>> users(scratch.userData.data(), count);
    std::shared_ptr<DrawRecordHit> hit;
    std::size_t fetched = 0;
    std::size_t index = 0;
    std::optional<Counter> first;
    for (; index < records.size() && fetched < DrawRecordStore::Candidates; ++index) {
        const auto& record = records[index];
        auto outcome = PrecheckDrawRecord(job, *record, users);
        if (outcome == Counter::Hits) {
            ++fetched;
            if (hit == nullptr) hit = std::make_shared<DrawRecordHit>();
            try {
                outcome = fetchDrawRecord(job, *record, *hit);
            } catch (const std::exception&) {
                outcome = Counter::MissFetchError;
            }
            if (outcome == Counter::Hits) {
                hit->record = record;
                checked = index + 1;
                return hit;
            }
        }
        if (index == 0) first = outcome;
    }
    checked = index;
    DrawRecords::Add(first.value_or(Counter::MissNoFetchPlan));
    return nullptr;
}

namespace {

struct HitRelease {
    std::shared_ptr<DrawRecordHit> hit;
    std::shared_ptr<VulkanDevice> device;
    std::shared_ptr<const DrawDecode> decode;

    HitRelease(std::shared_ptr<DrawRecordHit> kept, std::shared_ptr<VulkanDevice> keptDevice, std::shared_ptr<const DrawDecode> keptDecode) : hit(std::move(kept)), device(std::move(keptDevice)), decode(std::move(keptDecode)) {}
    HitRelease(const HitRelease&) = default;
    HitRelease(HitRelease&&) noexcept = default;
    HitRelease& operator=(const HitRelease&) = delete;
    HitRelease& operator=(HitRelease&&) = delete;

    ~HitRelease() {
        if (hit == nullptr && device == nullptr && decode == nullptr) return;
        std::array<std::shared_ptr<const void>, 3> held{std::move(hit), std::move(device), std::move(decode)};
        if (DrawFinisher::DeferRelease(held)) return;
        for (auto& entry : held) entry.reset();
    }
};

}

void Driver::commitRecordDraw(const LeanDrawJob& job, const LeanAnchor& anchor, std::shared_ptr<DrawRecordHit> hit) {
    auto writes = hit->record->writes;
    auto reads = std::move(hit->reads);
    const auto& graphics = job.shape->decode->state;
    const VulkanDevice::DrawTraceContext trace{0, job.serial, job.queue, static_cast<std::size_t>(job.packetOffset), graphics.hasColorTarget ? graphics.color.address : 0};
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectHandoff)) {
        auto& held = *hit;
        held.device = anchor.device;
        held.decode = job.shape->decode;
        held.trace = trace;
        held.epoch = job.epoch;
        held.shape = job.shapeKey;
        held.queue = job.queue;
        held.opcode = job.opcode;
        auto lean = [this, at = hit.get()] {
            auto& kept = *at;
            DrawPipeline::FollowEpoch(kept.epoch);
            GuestMemory::SetCurrentPacket(kept.opcode, kept.queue);
            const bool following = kept.device != nullptr && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::RunFollow);
            const bool attempted = following && Graphics::ExpressRunLed();
            static std::uint32_t itemOrdinal = 0;
            const bool timed = following && (++itemOrdinal & 7u) == 0;
            const auto began = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (attempted && followRecord(*kept.device, kept.queue, kept.decode->state, kept, kept.trace)) return;
            const auto committing = timed && attempted ? std::chrono::steady_clock::now() : began;
            commitRecord(kept.device, kept.queue, kept.decode->state, kept, kept.trace, kept.shape);
            if (timed) Graphics::NoteExpressNonFollower(attempted, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(committing - began).count()), static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - committing).count()));
            if (following) Graphics::DropExpressFollowVerify();
        };
        DirectDraws::Add(DirectDraws::Counter::HandoffsLean);
        DrawPipeline::Queue0().EnqueueKept(std::move(lean), std::move(writes), std::move(reads), std::move(hit));
        return;
    }
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::IdleHitRelease)) {
        auto released = [this, held = HitRelease{std::move(hit), anchor.device, job.shape->decode}, queue = job.queue, opcode = job.opcode, epoch = job.epoch, trace, shape = job.shapeKey] {
            DrawPipeline::FollowEpoch(epoch);
            GuestMemory::SetCurrentPacket(opcode, queue);
            const bool following = held.device != nullptr && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::RunFollow);
            const bool attempted = following && Graphics::ExpressRunLed();
            static std::uint32_t itemOrdinal = 0;
            const bool timed = following && (++itemOrdinal & 7u) == 0;
            const auto began = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (attempted && followRecord(*held.device, queue, held.decode->state, *held.hit, trace)) return;
            const auto committing = timed && attempted ? std::chrono::steady_clock::now() : began;
            commitRecord(held.device, queue, held.decode->state, *held.hit, trace, shape);
            if (timed) Graphics::NoteExpressNonFollower(attempted, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(committing - began).count()), static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - committing).count()));
            if (following) Graphics::DropExpressFollowVerify();
        };
        DrawPipeline::Queue0().Enqueue(std::move(released), std::move(writes), 0, {}, std::move(reads), true);
        return;
    }
    auto commit = [this, hit = std::move(hit), localDevice = anchor.device, decode = job.shape->decode, queue = job.queue, opcode = job.opcode, epoch = job.epoch, trace, shape = job.shapeKey] {
        DrawPipeline::FollowEpoch(epoch);
        GuestMemory::SetCurrentPacket(opcode, queue);
        const bool following = localDevice != nullptr && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::RunFollow);
        const bool attempted = following && Graphics::ExpressRunLed();
        static std::uint32_t itemOrdinal = 0;
        const bool timed = following && (++itemOrdinal & 7u) == 0;
        const auto began = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (attempted && followRecord(*localDevice, queue, decode->state, *hit, trace)) return;
        const auto committing = timed && attempted ? std::chrono::steady_clock::now() : began;
        commitRecord(localDevice, queue, decode->state, *hit, trace, shape);
        if (timed) Graphics::NoteExpressNonFollower(attempted, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(committing - began).count()), static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - committing).count()));
        if (following) Graphics::DropExpressFollowVerify();
    };
    DrawPipeline::Queue0().Enqueue(std::move(commit), std::move(writes), 0, {}, std::move(reads), true);
}

bool Driver::commitTicketDraw(const LeanDrawJob& job, const LeanAnchor& anchor, DrawRecordHit& draft, TicketArena& arena) {
    auto refusal = TicketFallback::Count;
    auto* const ticket = FlattenTicket(arena, draft, job, anchor.device, refusal);
    if (ticket == nullptr) {
        DrawTickets::AddFallback(refusal);
        if (draft.owner != nullptr) draft.record = static_cast<const DirectClass*>(draft.owner.get())->record;
        const auto* const drafted = &draft.parameters;
        auto hit = std::make_shared<DrawRecordHit>(std::move(draft));
        if (hit->preparedInputs.has_value() && hit->preparedInputs->draw == drafted) hit->preparedInputs->draw = &hit->parameters;
        commitRecordDraw(job, anchor, std::move(hit));
        return false;
    }
    struct Abandon {
        DrawTicket* held;
        ~Abandon() {
            if (held != nullptr) AbandonTicket(held);
        }
    } abandon{ticket};
    auto writes = draft.record->writes;
    DrawPipeline::Commit commit = [this, ticket] { commitTicket(ticket); };
    const auto reads = ticket->Reads();
    const auto lines = (ticket->followerBytes + 63u) / 64u;
    abandon.held = nullptr;
    DrawPipeline::Queue0().EnqueueTicket(std::move(commit), std::move(writes), reads, ticket, lines);
    return true;
}

void Driver::commitTicket(void* held) {
    auto& ticket = *static_cast<DrawTicket*>(held);
    DrawPipeline::FollowEpoch(ticket.epoch);
    GuestMemory::SetCurrentPacket(ticket.opcode, ticket.queue);
    const bool following = ticket.device != nullptr && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::RunFollow);
    const bool attempted = following && Graphics::ExpressRunLed();
    static std::uint32_t itemOrdinal = 0;
    const bool timed = following && (++itemOrdinal & 7u) == 0;
    const auto began = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (attempted) {
        const bool prefetched = DrawPipeline::TicketPrefetched();
        std::array<Graphics::KeptOwner, 2> recomputed{};
        const auto owners = FollowerOwners(ticket, recomputed);
        const auto buffers = ticket.BuffersView();
        const auto inputs = ticket.InputsView();
        const FollowParts parts{true, ticket.trace.token, *ticket.commit, ticket.parameters, ticket.Stages(), ticket.Has(DrawTicket::Buffers) ? &buffers : nullptr, ticket.Has(DrawTicket::Inputs) ? &inputs : nullptr, ticket.Has(DrawTicket::Fragment) ? &ticket.fragment : nullptr, owners, prefetched ? Graphics::FollowFormat::PrefetchedTicket : Graphics::FollowFormat::Ticket};
        if (followCore(*ticket.device, ticket.queue, *ticket.state, parts)) {
            DrawTickets::NoteCommit(true, prefetched);
            return;
        }
    }
    const auto committing = timed && attempted ? std::chrono::steady_clock::now() : began;
    {
        ThawedTicket thawed{ticket};
        commitRecord(thawed.TakeDevice(), ticket.queue, *ticket.state, thawed.Hit(), ticket.trace, ticket.shape);
    }
    if (timed) Graphics::NoteExpressNonFollower(attempted, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(committing - began).count()), static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - committing).count()));
    DrawTickets::NoteCommit(false, false);
    if (following) Graphics::DropExpressFollowVerify();
}

namespace {

DrawRecords::Counter CommitMiss(Graphics::DrawRecipeMiss miss) {
    using DrawRecords::Counter;
    switch (miss) {
    case Graphics::DrawRecipeMiss::NotRecordable: return Counter::CommitNotRecordable;
    case Graphics::DrawRecipeMiss::TargetGone: return Counter::CommitTargets;
    case Graphics::DrawRecipeMiss::TemplateGone: return Counter::CommitTemplate;
    case Graphics::DrawRecipeMiss::Proof: return Counter::CommitProof;
    case Graphics::DrawRecipeMiss::Moved: return Counter::CommitMoved;
    default: return Counter::CommitObjects;
    }
}

bool SameBoundBuffer(const Graphics::ShaderResources::BoundBuffer& a, const Graphics::ShaderResources::BoundBuffer& b) {
    if (a.set != b.set || a.binding != b.binding || a.element != b.element || a.snapshot != b.snapshot || a.info.range != b.info.range) return false;
    if (!a.snapshot) return a.info.buffer == b.info.buffer && a.info.offset == b.info.offset;
    return a.address == b.address && a.bytes == b.bytes && a.content == b.content;
}

std::string boundText(const Graphics::ShaderResources::BoundBuffer& entry) {
    const auto at = "set " + std::to_string(entry.set) + " binding " + std::to_string(entry.binding) + "[" + std::to_string(entry.element) + "] range " + std::to_string(entry.info.range);
    if (!entry.snapshot) return at + " from the template (buffer " + hex(reinterpret_cast<std::uint64_t>(entry.info.buffer)) + " at " + std::to_string(entry.info.offset) + ")";
    if (entry.address == 0) return at + " from " + std::to_string(entry.bytes) + " data bytes hashing to " + hex(entry.content);
    return at + " from " + std::to_string(entry.bytes) + " bytes at " + hex(entry.address);
}

std::string accessText(const DrawRecordAccess& access) {
    return std::to_string(access.reads.size()) + " reads, " + std::to_string(access.writes.size()) + " writes, " + std::to_string(access.writeOnly.size()) + " write-only ranges, " + std::to_string(access.images.size()) + " storage images" + (access.readsTarget ? ", reading a target" : "");
}

std::string lookupText(const DrawCommitView& today, const void* recorded) {
    const std::string which = today.found == recorded ? "the same template" : "the template it found";
    switch (today.lookup) {
    case DrawCommitLookup::Absent: return "found no template under the key";
    case DrawCommitLookup::Invalid: return "failed to revalidate " + which;
    case DrawCommitLookup::Unmoved: return "refused to move the read-only buffers of " + which;
    case DrawCommitLookup::Uncacheable: return "found the draw uncacheable";
    default: return "never looked a template up";
    }
}

static_assert(static_cast<std::size_t>(StagePlans::Counter::ProofUnsettled) - static_cast<std::size_t>(StagePlans::Counter::ProofsSkipped) + 1 == static_cast<std::size_t>(Graphics::ShaderResources::CleanProof::Count), "every clean proof outcome needs a stage plan counter");
static_assert(static_cast<std::size_t>(StagePlans::Counter::TargetsChanged) - static_cast<std::size_t>(StagePlans::Counter::TargetsHeld) + 1 == static_cast<std::size_t>(DrawRecordHold::Count), "every target hold outcome needs a stage plan counter");

void NotePlannedCommit(const DrawRecordCommit& commit) {
    using StagePlans::Counter;
    const auto& effects = commit.effects;
    if (effects.hold != DrawRecordHold::Count) StagePlans::AddOnCommitter(static_cast<Counter>(static_cast<std::size_t>(Counter::TargetsHeld) + static_cast<std::size_t>(effects.hold)));
    if (effects.cleanProof.has_value()) StagePlans::AddOnCommitter(static_cast<Counter>(static_cast<std::size_t>(Counter::ProofsSkipped) + static_cast<std::size_t>(*effects.cleanProof)));
    else if (effects.proved) StagePlans::AddOnCommitter(Counter::ProofsFull);
    if (effects.movedChecked != 0) StagePlans::AddOnCommitter(Counter::MovedChecked, effects.movedChecked);
    if (effects.movedTrusted != 0) StagePlans::AddOnCommitter(Counter::MovedTrusted, effects.movedTrusted);
}

void NoteDirectCommit(const DrawRecordCommit& commit, bool recorded) {
    using DirectDraws::Counter;
    DirectDraws::Add(recorded ? Counter::CommitHits : Counter::CommitMisses);
    const auto& effects = commit.effects;
    if (effects.cleanProof.has_value()) DirectDraws::Add(*effects.cleanProof == Graphics::ShaderResources::CleanProof::Skipped ? Counter::ProofsSkipped : Counter::ProofsNotSkipped);
    else if (effects.proved) DirectDraws::Add(Counter::ProofsFull);
    if (effects.sharedHold) DirectDraws::Add(Counter::HoldShared);
    else if (effects.hold == DrawRecordHold::Held) DirectDraws::Add(Counter::HoldOwn);
    else if (effects.hold != DrawRecordHold::Count) DirectDraws::Add(static_cast<Counter>(static_cast<std::size_t>(Counter::HoldFirst) + static_cast<std::size_t>(effects.hold) - static_cast<std::size_t>(DrawRecordHold::First)));
}

void NoteExpressVerify(const DrawRecord& record, DrawRecordCommit& commit, const Graphics::DrawRecipeOutcome& outcome, const std::shared_ptr<DrawExpressKeep>& keep, const std::vector<std::uint64_t>& versions, bool reprove) {
    using DirectDraws::Counter;
    using CleanProof = Graphics::ShaderResources::CleanProof;
    DirectDraws::Add(Counter::ExpressVerifyChecked);
    const char* what = nullptr;
    const auto proof = commit.effects.cleanProof;
    if (reprove) what = nullptr;
    else if (!outcome.recorded && outcome.miss == Graphics::DrawRecipeMiss::Proof) what = "the template proof";
    else if (proof.has_value() && (*proof == CleanProof::Rewritten || *proof == CleanProof::Forgotten || *proof == CleanProof::PendingWrites || *proof == CleanProof::Failed)) what = "the template's memory";
    else if (!outcome.recorded && outcome.miss == Graphics::DrawRecipeMiss::TargetGone) what = "a target";
    for (std::size_t index = 0; what == nullptr && keep != nullptr && index < keep->targets.size() && index < versions.size(); ++index) {
        if (keep->targets[index] != nullptr && keep->targets[index]->UploadVersion() != versions[index]) what = "a target's content";
    }
    if (what == nullptr) return;
    DirectDraws::Add(Counter::ExpressVerifyDiffering);
    commit.expressRefused = true;
    static std::atomic<std::int64_t> window{0};
    static std::atomic<std::uint32_t> printed{0};
    const auto now = static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10);
    if (window.exchange(now, std::memory_order_relaxed) != now) printed.store(0, std::memory_order_relaxed);
    if (printed.fetch_add(1, std::memory_order_relaxed) < 8) std::fprintf(stderr, "[express-commits-verify] %s differs: record %s; the express path is off for it\n", what, hex(record.probe).c_str());
}

void NoteCommitFragments(const DrawRecord& record, DrawRecordCommit& commit, bool timed, std::uint64_t nanoseconds) {
    using DirectDraws::Counter;
    const auto& note = commit.effects.fragment;
    if (note.expressCapture != DrawExpressCapture::None && note.expressCapture != DrawExpressCapture::Count) DirectDraws::Add(static_cast<Counter>(static_cast<std::size_t>(Counter::ExpressCaptured) + static_cast<std::size_t>(note.expressCapture) - static_cast<std::size_t>(DrawExpressCapture::Captured)));
    if (note.blockMiss != DrawFragmentBlockMiss::None && note.blockMiss != DrawFragmentBlockMiss::Count) DirectDraws::Add(static_cast<Counter>(static_cast<std::size_t>(Counter::FragmentsBlockNotReusable) + static_cast<std::size_t>(note.blockMiss) - static_cast<std::size_t>(DrawFragmentBlockMiss::NotReusable)));
    if (note.spliced) DirectDraws::Add(Counter::FragmentsSpliced);
    if (note.replayed) DirectDraws::Add(Counter::FragmentsReplayed);
    if (note.offered && !note.spliced && !note.replayed) {
        const auto drop = note.drop == DrawFragmentDrop::None || note.drop == DrawFragmentDrop::Count ? DrawFragmentDrop::Modes : note.drop;
        DirectDraws::Add(static_cast<Counter>(static_cast<std::size_t>(Counter::FragmentsDropObjects) + static_cast<std::size_t>(drop) - static_cast<std::size_t>(DrawFragmentDrop::Objects)));
    }
    if (note.verified) {
        DirectDraws::Add(Counter::FragmentsVerifyChecked);
        if (note.differs) {
            DirectDraws::Add(Counter::FragmentsVerifyDiffering);
            commit.fragmentsRefused = true;
            static std::atomic<std::int64_t> window{0};
            static std::atomic<std::uint32_t> printed{0};
            const auto now = static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10);
            if (window.exchange(now, std::memory_order_relaxed) != now) printed.store(0, std::memory_order_relaxed);
            if (printed.fetch_add(1, std::memory_order_relaxed) < 8) std::fprintf(stderr, "[commit-fragments-verify] %s differs: record %s; fragments are off for it\n", note.what != nullptr ? note.what : "a command", hex(record.probe).c_str());
        }
    }
    if (!timed) return;
    if (note.spliced || note.replayed) {
        DirectDraws::Add(Counter::FragmentsSplicedNanoseconds, nanoseconds);
        DirectDraws::Add(Counter::FragmentsSplicedTimed);
        return;
    }
    DirectDraws::Add(Counter::FragmentsTodayNanoseconds, nanoseconds);
    DirectDraws::Add(Counter::FragmentsTodayTimed);
}

// A hit that carries prepared buffers counts once: used, not used by reason (dropped when its
// commit did not get as far as the prepared buffers), or verified.
void NoteDirectBuffers(const DrawRecordCommit& commit, bool prepared) {
    using DirectDraws::Counter;
    const auto& effects = commit.effects;
    switch (effects.buffers) {
    case DrawRecordBuffers::None:
        if (prepared) DirectDraws::Add(Counter::BuffersDropped);
        break;
    case DrawRecordBuffers::Used: DirectDraws::Add(Counter::BuffersUsed); break;
    case DrawRecordBuffers::PlanDiffers: DirectDraws::Add(Counter::BuffersPlanDiffers); break;
    case DrawRecordBuffers::MovedHazard: DirectDraws::Add(Counter::BuffersMovedHazard); break;
    case DrawRecordBuffers::UnmovedHazard: DirectDraws::Add(Counter::BuffersUnmovedHazard); break;
    case DrawRecordBuffers::DepthGuards: DirectDraws::Add(Counter::BuffersDepthGuards); break;
    case DrawRecordBuffers::Checked: DirectDraws::Add(Counter::BuffersVerifyChecked); break;
    case DrawRecordBuffers::Differs:
        DirectDraws::Add(Counter::BuffersVerifyChecked);
        DirectDraws::Add(Counter::BuffersVerifyDiffering);
        DirectDraws::NoteBufferDifference(effects.bufferDifference.what, effects.bufferDifference.detail);
        break;
    case DrawRecordBuffers::MemoryChanged:
        DirectDraws::Add(Counter::BuffersVerifyChecked);
        DirectDraws::Add(Counter::BuffersMemoryChanged);
        break;
    case DrawRecordBuffers::Unwatched:
        DirectDraws::Add(Counter::BuffersVerifyChecked);
        DirectDraws::Add(Counter::BuffersUnwatched);
        DirectDraws::NoteBufferDifference(effects.bufferDifference.what, effects.bufferDifference.detail);
        break;
    default: break;
    }
    if (effects.bufferShortcuts == 0) return;
    const auto kind = effects.buffers == DrawRecordBuffers::Checked ? 0u : effects.buffers == DrawRecordBuffers::Differs ? 1u : effects.buffers == DrawRecordBuffers::MemoryChanged ? 2u : effects.buffers == DrawRecordBuffers::Unwatched ? 3u : 4u;
    if (kind == 4u) return;
    using Prepared = Graphics::ShaderResources::PreparedBuffers;
    constexpr std::array<std::pair<std::uint8_t, Counter>, 3> shortcuts{{{Prepared::DeferredImports, Counter::VerifyDeferredChecked}, {Prepared::MergedSettle, Counter::VerifyMergedChecked}, {Prepared::ExactCapacity, Counter::VerifyCapacityChecked}}};
    for (const auto& [bit, first] : shortcuts) {
        if ((effects.bufferShortcuts & bit) == 0) continue;
        DirectDraws::Add(first);
        if (kind != 0u) DirectDraws::Add(static_cast<Counter>(static_cast<std::size_t>(first) + kind));
    }
}

void NoteDirectInputs(const DrawRecordCommit& commit, bool prepared) {
    using DirectDraws::Counter;
    switch (commit.effects.inputs) {
    case DrawRecordInputs::None:
        if (prepared) DirectDraws::Add(Counter::InputsDropped);
        break;
    case DrawRecordInputs::Used: DirectDraws::Add(Counter::InputsUsed); break;
    case DrawRecordInputs::Rechecked:
        DirectDraws::Add(Counter::InputsUsed);
        DirectDraws::Add(Counter::InputsRechecked);
        break;
    case DrawRecordInputs::ImportChanged: DirectDraws::Add(Counter::InputsImportChanged); break;
    case DrawRecordInputs::Shape: DirectDraws::Add(Counter::InputsCommitShape); break;
    case DrawRecordInputs::Refused: DirectDraws::Add(Counter::InputsRefused); break;
    case DrawRecordInputs::Checked: DirectDraws::Add(Counter::InputsVerifyChecked); break;
    case DrawRecordInputs::Differs:
        DirectDraws::Add(Counter::InputsVerifyChecked);
        DirectDraws::Add(Counter::InputsVerifyDiffering);
        break;
    default: break;
    }
    if (!commit.effects.inputsCapacity || (commit.effects.inputs != DrawRecordInputs::Checked && commit.effects.inputs != DrawRecordInputs::Differs)) return;
    DirectDraws::Add(Counter::InputsVerifyCapacityChecked);
    if (commit.effects.inputs == DrawRecordInputs::Differs) DirectDraws::Add(Counter::InputsVerifyCapacityDiffering);
}

void PublishInputsRecipe(DrawRecordCommit& commit) {
    if (commit.inputsRecipeAsked || commit.recipe == nullptr || !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectInputs)) return;
    commit.inputsRecipeAsked = true;
    commit.inputsRecipe.store(commit.recipe);
}

void PublishBufferPlan(DrawRecordCommit& commit, std::span<const Graphics::CompiledShader> stages) {
    if (commit.bufferPlanAsked || !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectBuffers)) return;
    commit.bufferPlanAsked = true;
    if (const auto resources = commit.recipe->templateRef.lock(); resources != nullptr) commit.bufferPlan.store(resources->DirectBufferPlan(stages));
}

bool SameVertexInput(const Graphics::VertexInputLayout& a, const Graphics::VertexInputLayout& b) {
    const auto sameBinding = [](const VkVertexInputBindingDescription& x, const VkVertexInputBindingDescription& y) { return x.binding == y.binding && x.stride == y.stride && x.inputRate == y.inputRate; };
    const auto sameAttribute = [](const VkVertexInputAttributeDescription& x, const VkVertexInputAttributeDescription& y) { return x.location == y.location && x.binding == y.binding && x.format == y.format && x.offset == y.offset; };
    return std::ranges::equal(a.bindings, b.bindings, sameBinding) && std::ranges::equal(a.attributes, b.attributes, sameAttribute);
}

using CommitDifferenceNote = void (*)(DrawRecords::Field, std::string_view);
using CommitAllowedNote = void (*)(DrawRecords::Allowed);
bool compareCommitViews(const DrawRecord& record, const DrawCommitView& answer, const DrawCommitView& today, CommitDifferenceNote noteDifference, CommitAllowedNote noteAllowed, bool corrupting);

void noteRepeatDifference(DrawRecords::Field field, std::string_view detail) {
    DirectDraws::NoteRepeatDifference(DrawRecords::FieldName(field), detail);
}

void noteRepeatAllowed(DrawRecords::Allowed) {}

void noteBindingsDifference(DrawRecords::Field field, std::string_view detail) {
    DirectDraws::NoteBindingsDifference(DrawRecords::FieldName(field), detail);
}

void NoteCommitBindings(const DrawRecord& record, DrawRecordCommit& commit, const Graphics::ShaderResources::PreparedBuffers* prepared) {
    using DirectDraws::Counter;
    using Taken = Graphics::ShaderResources::BindingTaken;
    auto& noted = commit.effects.bindings;
    if (prepared != nullptr && prepared->placed != nullptr) DirectDraws::Add(Counter::BindingsReturned);
    if (noted.timed && (noted.pushed || noted.written)) {
        const bool placed = noted.taken == Taken::Placed;
        const auto nanoseconds = noted.written ? (placed ? Counter::BindingsNewWrittenNanoseconds : Counter::BindingsTodayWrittenNanoseconds) : (placed ? Counter::BindingsNewPushedNanoseconds : Counter::BindingsTodayPushedNanoseconds);
        DirectDraws::Add(nanoseconds, noted.nanoseconds);
        DirectDraws::Add(static_cast<Counter>(static_cast<std::size_t>(nanoseconds) + 1u));
    }
    if (noted.timed && noted.written) {
        const auto counter = noted.dynamicBound != 0 ? (noted.pushed ? Counter::BindingsDynamicPushedNanoseconds : Counter::BindingsDynamicUnpushedNanoseconds) : (noted.pushed ? Counter::BindingsWrittenPushedNanoseconds : Counter::BindingsWrittenUnpushedNanoseconds);
        DirectDraws::Add(counter, noted.nanoseconds);
        DirectDraws::Add(static_cast<Counter>(static_cast<std::size_t>(counter) + 1u));
    }
    if (noted.timed && noted.keepNanoseconds != 0) {
        DirectDraws::Add(Counter::BindingsKeepNanoseconds, noted.keepNanoseconds);
        DirectDraws::Add(Counter::BindingsKeepTimed);
    }
    if (noted.dynamicLeave != 0 && noted.written && noted.dynamicBound == 0) DirectDraws::Add(static_cast<Counter>(static_cast<std::size_t>(noted.pushed ? Counter::BindingsLeftPushedRefused : Counter::BindingsLeftUnpushedRefused) + noted.dynamicLeave - 1u));
    if (!noted.routed) return;
    switch (noted.taken) {
    case Taken::Placed:
        DirectDraws::Add(Counter::BindingsTaken);
        if (noted.written) DirectDraws::Add(Counter::BindingsTakenWritten);
        DirectDraws::Add(Counter::BindingsBytes, noted.bytes);
        DirectDraws::Add(Counter::BindingsObjects, noted.objects);
        break;
    case Taken::Copied: DirectDraws::Add(Counter::BindingsCopied); break;
    case Taken::OtherDevice: DirectDraws::Add(Counter::BindingsOtherDevice); break;
    default:
        if (commit.effects.buffers == DrawRecordBuffers::UnmovedHazard) DirectDraws::Add(Counter::BindingsHazard);
        else if (commit.effects.buffers == DrawRecordBuffers::DepthGuards) DirectDraws::Add(Counter::BindingsDepthGuards);
        break;
    }
    if (noted.tested != 0) DirectDraws::Add(Counter::BindingsTested, noted.tested);
    if (noted.stamped != 0) DirectDraws::Add(Counter::BindingsStamped, noted.stamped);
    if (noted.guards != 0) DirectDraws::Add(Counter::BindingsGuards, noted.guards);
    if (noted.dynamicBound != 0) DirectDraws::Add(noted.pushed ? Counter::BindingsDynamicPushedBound : Counter::BindingsDynamicBound, noted.dynamicBound);
    if (noted.dynamicHits != 0) DirectDraws::Add(Counter::BindingsDynamicHits, noted.dynamicHits);
    if (noted.dynamicWritten != 0) DirectDraws::Add(Counter::BindingsDynamicWritten, noted.dynamicWritten);
    if (noted.dynamicRefused != 0) DirectDraws::Add(Counter::BindingsDynamicRefused, noted.dynamicRefused);
    if (noted.setsLent != 0) DirectDraws::Add(Counter::BindingsSetsLent, noted.setsLent);
    if (noted.setsWritten != 0) DirectDraws::Add(Counter::BindingsSetsWritten, noted.setsWritten);
    bool differs = false;
    if (noted.refuted != 0) {
        DirectDraws::Add(Counter::BindingsRefuted, noted.refuted);
        DirectDraws::NoteBindingsDifference("stamp claim", keyText(record.key) + ": an unmoved range the stamp held clear has a pending write");
        differs = true;
    }
    if (noted.checked) {
        DirectDraws::Add(Counter::BindingsVerifyChecked);
        DrawCommitView mine;
        DrawCommitView today;
        mine.filled = today.filled = true;
        mine.bound = today.bound = true;
        mine.bindings = std::move(noted.mine);
        today.bindings = std::move(noted.today);
        if (compareCommitViews(record, mine, today, &noteBindingsDifference, &noteRepeatAllowed, false)) differs = true;
        if (noted.taken == Taken::Placed && !noted.todayTaken) {
            DirectDraws::NoteBindingsDifference("stamp claim", keyText(record.key) + ": today's tests refused the prepared buffers the finisher's bindings were bound for");
            differs = true;
        }
    }
    if (!differs) return;
    DirectDraws::Add(Counter::BindingsVerifyDiffering);
    commit.bindingsRefused = true;
}

void noteRepeatRefusal(const Graphics::RepeatOutcome& repeated) {
    using DirectDraws::Counter;
    if (repeated.refusal == Graphics::RepeatRefusal::Count) return;
    DirectDraws::Add(static_cast<Counter>(static_cast<std::size_t>(Counter::RepeatNoBlock) + static_cast<std::size_t>(repeated.refusal)));
    if (repeated.refusal == Graphics::RepeatRefusal::NotSimple && repeated.simple != DrawRepeatClass::Simple) DirectDraws::Add(static_cast<Counter>(static_cast<std::size_t>(Counter::RepeatClassNotRecorded) + static_cast<std::size_t>(repeated.simple) - 1u));
    if (repeated.refusal == Graphics::RepeatRefusal::NotSimple && repeated.detail != DrawRepeatDetail::None) DirectDraws::Add(static_cast<Counter>(static_cast<std::size_t>(Counter::RepeatDetailNotReusable) + static_cast<std::size_t>(repeated.detail) - 1u));
    if (repeated.late) DirectDraws::Add(Counter::RepeatLate);
}

bool RepeatTimings() {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    return stats;
}

void verifyRepeat(VulkanDevice& device, const Graphics::State& graphics, DrawRecordHit& hit, DrawRecordCommit& commit, const VulkanDevice::DrawTraceContext* trace, const Graphics::ShaderResources::PreparedBuffers* prepared, Graphics::PreparedInputs* inputs) {
    using DirectDraws::Counter;
    DrawCommitView mine;
    const auto described = device.RepeatFromRecord(graphics, hit.parameters, hit.stages, commit, trace, prepared, inputs, &mine);
    if (!described.taken) return;
    DirectDraws::Add(Counter::RepeatVerifyChecked);
    DrawCommitView today;
    const auto answered = device.DrawFromRecord(graphics, hit.parameters, hit.stages, commit, trace, &today);
    const auto at = keyText(hit.record->key);
    bool refuted = false;
    const auto refute = [&](const std::string& claim, const std::string& detail) {
        DirectDraws::NoteRepeatDifference("stamp claim " + claim, at + ": " + detail);
        refuted = true;
    };
    if (described.refuted != nullptr) refute(described.refuted, "a range the stamp holds clear has a pending write");
    if (answered.miss != Graphics::DrawRecipeMiss::None) {
        if (commit.effects.hold == DrawRecordHold::Count) refute(Graphics::DrawRecipeMissName(answered.miss), "today's way missed before its hold although the stamp held");
        else if (answered.miss == Graphics::DrawRecipeMiss::Proof && described.proofSkipped) refute("proof", "today's proof failed although the template's proof memo held");
    } else if (described.shared ? !commit.effects.sharedHold : commit.effects.hold != DrawRecordHold::Held) {
        refute("hold", described.shared ? "today's way does not hold the pass under the shared pass memo" : "today's way does not hold the record's own pass memo");
    } else if (!today.keptOnce || (!described.shared && !today.access.has_value())) {
        refute(today.keptOnce ? "pass access" : "keep", "today's way does not take the record's per-batch keep or pass access memo");
    }
    if (refuted) DirectDraws::Add(Counter::RepeatRefuted);
    bool differs = compareCommitViews(*hit.record, mine, today, &noteRepeatDifference, &noteRepeatAllowed, false);
    if (described.pushDifference.what != nullptr) {
        DirectDraws::NoteRepeatDifference(described.pushDifference.what, at + ": " + described.pushDifference.detail);
        differs = true;
    }
    if (differs) DirectDraws::Add(Counter::RepeatVerifyDiffering);
    if (differs || refuted) commit.repeat.reset();
}

struct FollowTally {
    std::uint64_t followed = 0;
};
struct FollowTallyStorage {};

FollowTally& followTally() {
    return HostThreadLocal<FollowTally, FollowTallyStorage>();
}

void flushFollowTally(FollowTally& tally) {
    const auto count = tally.followed;
    if (count == 0) return;
    tally.followed = 0;
    DirectDraws::Add(DirectDraws::Counter::ExpressTaken, count);
    DirectDraws::Add(DirectDraws::Counter::FragmentsSpliced, count);
    DirectDraws::Add(DirectDraws::Counter::CommitHits, count);
    DirectDraws::Add(DirectDraws::Counter::ProofsSkipped, count);
    DirectDraws::Add(DirectDraws::Counter::HoldOwn, count);
    DirectDraws::Add(DirectDraws::Counter::BuffersUsed, count);
    DirectDraws::Add(DirectDraws::Counter::InputsUsed, count);
    DrawRecords::AddOnCommitter(DrawRecords::Counter::CommitHits, count);
    DrawRecords::AddOnCommitter(DrawRecords::Counter::CommitPassHeld, count);
    DrawRecords::AddOnCommitter(DrawRecords::Counter::CommitKeptOnce, count);
}

struct CommitOrdinals {
    std::uint64_t record = 0;
    std::uint64_t buffers = 0;
    std::uint64_t inputs = 0;
    std::uint64_t express = 0;
    std::uint64_t expressRefused = ~std::uint64_t{0};
};
CommitOrdinals commitOrdinals;

bool ordinalDue(std::uint64_t ordinal, std::uint64_t every) {
    return every != 0 && (ordinal + 1) % every == 0;
}

}

bool Driver::followRecord(VulkanDevice& localDevice, std::uint32_t queue, const Graphics::State& graphics, DrawRecordHit& hit, const VulkanDevice::DrawTraceContext& trace) {
    const auto buffers = hit.preparedBuffers.has_value() ? hit.preparedBuffers->View() : Graphics::ShaderResources::PreparedBuffersView{};
    const auto inputs = hit.preparedInputs.has_value() ? hit.preparedInputs->View() : Graphics::PreparedInputsView{};
    const FollowParts parts{hit.direct, trace.token, *hit.record->commit, hit.parameters, hit.stages, hit.preparedBuffers.has_value() ? &buffers : nullptr, hit.preparedInputs.has_value() ? &inputs : nullptr, hit.fragment.has_value() ? &*hit.fragment : nullptr, {}, Graphics::FollowFormat::Hit};
    return followCore(localDevice, queue, graphics, parts);
}

bool Driver::followCore(VulkanDevice& localDevice, std::uint32_t queue, const Graphics::State& graphics, const FollowParts& parts) {
    using Refusal = Graphics::ExpressRunRefusal;
    static std::uint32_t timingOrdinal = 0;
    const bool timed = (++timingOrdinal & 7u) == 0;
    const auto began = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto& tally = followTally();
    const auto refuse = [&tally](Refusal why) {
        flushFollowTally(tally);
        Graphics::RefuseExpressFollow(why);
        return false;
    };
    auto& commit = parts.commit;
    if (!parts.direct || parts.token != 0 || commit.recipe == nullptr || commit.writers != 1 || !commit.bufferPlanAsked || !commit.inputsRecipeAsked || parts.fragment == nullptr || parts.buffers == nullptr || parts.inputs == nullptr) return refuse(Refusal::NotDirect);
    static const bool verifies = DrawRecords::VerifyEvery() != 0 || DirectDraws::BufferVerifyEvery() != 0 || DirectDraws::InputVerifyEvery() != 0 || Graphics::CommitFragmentsVerifyEvery() != 0 || Graphics::ExpressVerifyEvery() != 0;
    auto& ordinals = commitOrdinals;
    bool expressDue = false;
    if (verifies) {
        if (ordinalDue(ordinals.record, DrawRecords::VerifyEvery()) || ordinalDue(ordinals.buffers, DirectDraws::BufferVerifyEvery()) || ordinalDue(ordinals.inputs, DirectDraws::InputVerifyEvery())) return refuse(Refusal::Verifies);
        expressDue = commit.express != nullptr && commit.express->valid && ordinalDue(ordinals.express, Graphics::ExpressVerifyEvery());
        if (expressDue && ordinals.expressRefused != ordinals.express) {
            ordinals.expressRefused = ordinals.express;
            return refuse(Refusal::Verifies);
        }
    }
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::XpNoCommits)) return refuse(Refusal::Experiment);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) return refuse(Refusal::Unlocked);
    if (!packetLabelsIdle()) return refuse(Refusal::Labels);
    commit.cleanProofs = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectCleanProofs);
    commit.verifyBuffers = false;
    commit.verifyInputs = false;
    const auto attempts = Graphics::ExpressFollowAttempts();
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FollowWider)) Graphics::NoteExpressFollowDue(expressDue);
    if (!localDevice.ExpressFollow(graphics, parts.parameters, parts.stages, commit, *parts.buffers, *parts.inputs, *parts.fragment, parts.owners, parts.format)) {
        if (Graphics::ExpressFollowAttempts() == attempts) return refuse(Refusal::Device);
        flushFollowTally(tally);
        return false;
    }
    if (++tally.followed >= 64) flushFollowTally(tally);
    if (verifies) {
        if (DrawRecords::VerifyEvery() != 0) ++ordinals.record;
        if (DirectDraws::BufferVerifyEvery() != 0) ++ordinals.buffers;
        if (DirectDraws::InputVerifyEvery() != 0) ++ordinals.inputs;
        if (Graphics::ExpressVerifyEvery() != 0 && !expressDue) ++ordinals.express;
    }
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::XpSlowCommitter)) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(1);
        while (std::chrono::steady_clock::now() < until) {}
    }
    ThreadReport::MarkItem(ThreadReport::ItemMark::Planned);
    submitDueAfterCommit(&localDevice);
    if (timed) Graphics::NoteExpressFollowItem(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - began).count()));
    static_cast<void>(queue);
    return true;
}

void Driver::commitRecord(std::shared_ptr<VulkanDevice> localDevice, std::uint32_t queue, const Graphics::State& graphics, DrawRecordHit& hit, const VulkanDevice::DrawTraceContext& trace, std::uint64_t shape) {
    using DrawRecords::Counter;
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::XpSlowCommitter)) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(1);
        while (std::chrono::steady_clock::now() < until) {}
    }
    if (hit.direct && hit.record->commit->recipe != nullptr && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::XpNoCommits)) {
        ThreadReport::MarkItem(ThreadReport::ItemMark::Planned);
        return;
    }
    if (Graphics::ExpressRunLed() && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::RunFollow)) {
        if (localDevice != nullptr && followRecord(*localDevice, queue, graphics, hit, trace)) return;
        Graphics::BreakExpressRun();
    }
    const auto* tracePtr = trace.token != 0 ? &trace : nullptr;
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    auto fragmentOutputs = hit.record->fragmentOutputs;
    if (auto current = device.Load(); current != nullptr && current != localDevice) {
        localDevice = std::move(current);
        fragmentOutputs.reset();
    }
    struct SubmitDue {
        Driver& driver;
        const std::shared_ptr<VulkanDevice>& device;
        ~SubmitDue() noexcept(false) {
            if (std::uncaught_exceptions() == 0) driver.submitDueAfterCommit(device.get());
        }
    } submitDue{*this, localDevice};
    recordLabelsForPacket(localDevice.get(), queue);
    auto& commit = *hit.record->commit;
    if (commit.writers == 0) commit.writers = StagesMayWriteBuffers(hit.stages) ? 2 : 1;
    if (commit.writers == 2) noteDrawWriters(hit.stages, queue);
    commit.cleanProofs = hit.direct ? Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectCleanProofs) : hit.planned && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoPlanCleanProofs);
    auto& ordinal = commitOrdinals.record;
    const auto every = DrawRecords::VerifyEvery();
    const bool verify = commit.recipe != nullptr && every != 0 && ++ordinal % every == 0;
    if (commit.recipe == nullptr) {
        DrawRecords::AddOnCommitter(Counter::CommitFirst);
        if (hit.direct) DirectDraws::Add(DirectDraws::Counter::CommitFirst);
    } else if (!verify) {
        const auto* prepared = hit.preparedBuffers.has_value() ? &*hit.preparedBuffers : nullptr;
        auto& bufferOrdinal = commitOrdinals.buffers;
        const auto bufferEvery = DirectDraws::BufferVerifyEvery();
        commit.verifyBuffers = prepared != nullptr && bufferEvery != 0 && ++bufferOrdinal % bufferEvery == 0;
        auto* inputs = hit.preparedInputs.has_value() ? &*hit.preparedInputs : nullptr;
        auto& inputOrdinal = commitOrdinals.inputs;
        const auto inputEvery = DirectDraws::InputVerifyEvery();
        commit.verifyInputs = inputs != nullptr && inputEvery != 0 && ++inputOrdinal % inputEvery == 0;
        // DrawFromRecord can return before DrawWithRecord resets them.
        commit.effects = {};
        std::optional<Graphics::DrawRecipeOutcome> repeatedOutcome;
        auto* fullInputs = inputs;
        auto takenInputs = DrawRecordInputs::None;
        const bool repeatOn = hit.direct && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectRepeat);
        static std::uint64_t timingOrdinal = 0;
        const bool timed = hit.direct && RepeatTimings() && (++timingOrdinal & 7u) == 0;
        if (repeatOn && !commit.verifyBuffers && !commit.verifyInputs) {
            using DirectDraws::Counter;
            const auto began = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const auto repeated = localDevice->RepeatFromRecord(graphics, hit.parameters, hit.stages, commit, tracePtr, prepared, inputs, nullptr, DirectDraws::RepeatVerifyEvery());
            if (repeated.taken) {
                if (timed) {
                    const auto nanoseconds = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - began).count());
                    DirectDraws::Add(Counter::RepeatNanoseconds, nanoseconds);
                    DirectDraws::Add(Counter::RepeatTimed);
                    if (repeated.sets) {
                        DirectDraws::Add(Counter::RepeatSetsNanoseconds, nanoseconds);
                        DirectDraws::Add(Counter::RepeatSetsTimed);
                    }
                    if (repeated.stores) {
                        DirectDraws::Add(Counter::RepeatStoresNanoseconds, nanoseconds);
                        DirectDraws::Add(Counter::RepeatStoresTimed);
                    }
                    if (repeated.shared) {
                        DirectDraws::Add(Counter::RepeatSharedNanoseconds, nanoseconds);
                        DirectDraws::Add(Counter::RepeatSharedTimed);
                    }
                }
                DirectDraws::Add(Counter::RepeatTaken);
                if (repeated.sets) DirectDraws::Add(Counter::RepeatTakenSets);
                if (repeated.stores) DirectDraws::Add(Counter::RepeatTakenStores);
                if (repeated.shared) DirectDraws::Add(Counter::RepeatTakenShared);
                if (repeated.passBegun) DirectDraws::Add(Counter::RepeatPassBegun);
                if (repeated.proofSkipped) DirectDraws::Add(Counter::RepeatProofsSkipped);
                if (repeated.unmovedSkipped) DirectDraws::Add(Counter::RepeatUnmovedSkipped);
                if (!repeated.result.recorded) DirectDraws::Add(Counter::RepeatProofMisses);
                repeatedOutcome = repeated.result;
            } else {
                noteRepeatRefusal(repeated);
                if (repeated.refusal == Graphics::RepeatRefusal::VerifyDue) verifyRepeat(*localDevice, graphics, hit, commit, tracePtr, prepared, inputs);
                else if (repeated.refusal == Graphics::RepeatRefusal::Inputs && repeated.inputs != DrawRecordInputs::None) {
                    fullInputs = nullptr;
                    takenInputs = repeated.inputs;
                }
                commit.effects = {};
            }
        }
        const auto fullBegan = timed && !repeatedOutcome.has_value() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const bool fragmentsOn = hit.direct && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::CommitFragments);
        static std::uint64_t fragmentTimingOrdinal = 0;
        const bool fragmentTimed = fragmentsOn && !repeatedOutcome.has_value() && (++fragmentTimingOrdinal & 7u) == 0;
        const auto fragmentBegan = fragmentTimed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        auto* fragment = fragmentsOn && hit.fragment.has_value() ? &*hit.fragment : nullptr;
        std::optional<Graphics::DrawRecipeOutcome> expressedOutcome;
        std::shared_ptr<DrawExpressKeep> verifyKeep;
        bool verifyReprove = false;
        std::vector<std::uint64_t> verifyVersions;
        if (fragment != nullptr && !repeatedOutcome.has_value() && tracePtr == nullptr && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::ExpressCommits)) {
            using DirectDraws::Counter;
            auto& expressOrdinal = commitOrdinals.express;
            const auto expressEvery = Graphics::ExpressVerifyEvery();
            const bool due = expressEvery != 0 && commit.express != nullptr && commit.express->valid && ++expressOrdinal % expressEvery == 0;
            const auto expressBegan = fragmentTimed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const auto expressed = localDevice->ExpressFromRecord(graphics, hit.parameters, hit.stages, commit, prepared, fullInputs, fragment, due);
            if (expressed.taken) {
                expressedOutcome = expressed.result;
                DirectDraws::Add(Counter::ExpressTaken);
                if (expressed.reproved) {
                    using ExpressMiss = Graphics::ShaderResources::ExpressMiss;
                    DirectDraws::Add(Counter::ExpressReproved);
                    if (expressed.reproveMiss >= ExpressMiss::NotProved && expressed.reproveMiss <= ExpressMiss::Staging) DirectDraws::Add(static_cast<Counter>(static_cast<std::size_t>(Counter::ExpressReprovedNotProved) + static_cast<std::size_t>(expressed.reproveMiss) - static_cast<std::size_t>(ExpressMiss::NotProved)));
                }
                if (expressed.refreshed != 0) DirectDraws::Add(Counter::ExpressRefreshed, expressed.refreshed);
                if (fragmentTimed) {
                    DirectDraws::Add(Counter::ExpressNanoseconds, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - expressBegan).count()));
                    DirectDraws::Add(Counter::ExpressTimed);
                }
            } else {
                const auto reason = expressed.miss == DrawExpressMiss::None || expressed.miss == DrawExpressMiss::Count ? DrawExpressMiss::Shape : expressed.miss;
                DirectDraws::Add(static_cast<Counter>(static_cast<std::size_t>(Counter::ExpressNoState) + static_cast<std::size_t>(reason) - static_cast<std::size_t>(DrawExpressMiss::NoState)));
                if (expressed.verify && commit.express != nullptr && commit.express->keep != nullptr) {
                    verifyKeep = commit.express->keep;
                    verifyReprove = expressed.reproved;
                    for (const auto& target : verifyKeep->targets) verifyVersions.push_back(target != nullptr ? target->UploadVersion() : 0u);
                }
            }
        }
        const auto outcome = expressedOutcome.has_value() ? *expressedOutcome : repeatedOutcome.has_value() ? *repeatedOutcome : localDevice->DrawFromRecord(graphics, hit.parameters, hit.stages, commit, tracePtr, nullptr, prepared, fullInputs, fragment);
        if (verifyKeep != nullptr) NoteExpressVerify(*hit.record, commit, outcome, verifyKeep, verifyVersions, verifyReprove);
        const bool captureUnreached = fragment != nullptr && !expressedOutcome.has_value() && !repeatedOutcome.has_value() && tracePtr == nullptr && commit.effects.fragment.expressCapture == DrawExpressCapture::None && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::ExpressCommits);
        if (captureUnreached) DirectDraws::Add(DirectDraws::Counter::ExpressCaptureUnreached);
        if (fragmentsOn && hit.parameters.indirect) {
            const auto capture = commit.effects.fragment.expressCapture;
            if (capture == DrawExpressCapture::Captured) DirectDraws::Add(DirectDraws::Counter::ExpressIndirectCaptured);
            else if (captureUnreached || (capture != DrawExpressCapture::None && capture != DrawExpressCapture::Count)) DirectDraws::Add(DirectDraws::Counter::ExpressIndirectNotCaptured);
        }
        if (fragmentsOn) NoteCommitFragments(*hit.record, commit, fragmentTimed && outcome.recorded, fragmentTimed ? static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - fragmentBegan).count()) : 0u);
        if (timed && !repeatedOutcome.has_value() && outcome.recorded && commit.effects.hold == DrawRecordHold::Held && !commit.effects.sharedHold) {
            DirectDraws::Add(DirectDraws::Counter::RepeatFullNanoseconds, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - fullBegan).count()));
            DirectDraws::Add(DirectDraws::Counter::RepeatFullTimed);
        }
        if (takenInputs != DrawRecordInputs::None) commit.effects.inputs = takenInputs;
        if (repeatOn && !repeatedOutcome.has_value() && commit.repeat != nullptr && commit.repeat->simple == DrawRepeatClass::Simple) DirectDraws::Add(DirectDraws::Counter::RepeatBlocks);
        if (hit.planned) NotePlannedCommit(commit);
        if (hit.direct) {
            NoteDirectCommit(commit, outcome.recorded);
            NoteDirectBuffers(commit, prepared != nullptr);
            NoteDirectInputs(commit, inputs != nullptr);
            NoteCommitBindings(*hit.record, commit, prepared);
        }
        if (outcome.recorded) {
            if (hit.direct) {
                PublishBufferPlan(commit, hit.stages);
                PublishInputsRecipe(commit);
            }
            if (hit.planned || hit.direct) ThreadReport::MarkItem(ThreadReport::ItemMark::Planned);
            const auto& effects = commit.effects;
            DrawRecords::AddOnCommitter(Counter::CommitHits);
            if (effects.pushed) DrawRecords::AddOnCommitter(Counter::CommitPushed);
            if (effects.unpushed) DrawRecords::AddOnCommitter(Counter::CommitUnpushed);
            if (effects.passHeld) DrawRecords::AddOnCommitter(Counter::CommitPassHeld);
            if (effects.keptOnce) DrawRecords::AddOnCommitter(Counter::CommitKeptOnce);
            if (effects.notesSkipped != 0) DrawRecords::AddOnCommitter(Counter::CommitNotesSkipped, effects.notesSkipped);
            if (effects.movedTrusted != 0) DrawRecords::AddOnCommitter(Counter::CommitMovedTrusted, effects.movedTrusted);
            return;
        }
        DrawRecords::AddOnCommitter(CommitMiss(outcome.miss));
    }
    if ((commit.recipe == nullptr || verify) && hit.preparedBuffers.has_value()) DirectDraws::Add(DirectDraws::Counter::BuffersDropped);
    if ((commit.recipe == nullptr || verify) && hit.preparedBuffers.has_value() && hit.preparedBuffers->placed != nullptr) DirectDraws::Add(DirectDraws::Counter::BindingsReturned);
    if ((commit.recipe == nullptr || verify) && hit.preparedInputs.has_value()) DirectDraws::Add(DirectDraws::Counter::InputsDropped);
    if (hit.planned || hit.direct) ThreadReport::MarkItem(ThreadReport::ItemMark::PlanFallback);
    DrawCommitView answer;
    if (verify) static_cast<void>(localDevice->DrawFromRecord(graphics, hit.parameters, hit.stages, commit, tracePtr, &answer));
    if (verify && DrawRecords::CorruptingCommits() && !answer.bindings.empty()) {
        const auto snapshot = std::ranges::find_if(answer.bindings, [](const Graphics::ShaderResources::BoundBuffer& entry) { return entry.snapshot; });
        (snapshot != answer.bindings.end() ? *snapshot : answer.bindings.front()).info.range ^= 4u;
    }
    DrawCommitView today;
    struct ViewScope {
        ~ViewScope() { DrawCommitViewSlot::Set(nullptr); }
    } viewScope;
    if (verify) {
        today.wantPush = answer.pushed;
        DrawCommitViewSlot::Set(&today);
    }
    std::shared_ptr<const DrawRecipe> built;
    const auto& record = *hit.record;
    localDevice->Draw(graphics, hit.parameters, hit.stages, hit.snapshots, &built, tracePtr, shape, record.contentKey.empty() ? nullptr : &record.contentKey, fragmentOutputs ? &*fragmentOutputs : nullptr, true);
    DrawCommitViewSlot::Set(nullptr);
    bool differs = false;
    if (verify) {
        DrawRecords::AddOnCommitter(Counter::CommitVerified);
        differs = verifyRecordCommit(record, answer, today);
    }
    if (differs) {
        commit.Adopt(nullptr);
        DrawRecords::AddOnCommitter(Counter::CommitDemoted);
        // One recipe serves every draw of a direct class, so a class whose commit differed once
        // goes back to today's path for good.
        if (hit.direct) RefuseDirectClass(hit.owner);
        return;
    }
    if (built != nullptr && (!verify || !answer.filled || answer.resources != today.resources)) {
        commit.Adopt(std::move(built));
        DrawRecords::AddOnCommitter(Counter::CommitBuilt);
    } else if (!verify) {
        DrawRecords::AddOnCommitter(Counter::CommitUnbuilt);
    }
}

bool Driver::verifyRecordCommit(const DrawRecord& record, const DrawCommitView& answer, const DrawCommitView& today) {
    DrawRecords::NoteChecked(DrawRecords::Item::Commit);
    return compareCommitViews(record, answer, today, &DrawRecords::NoteDifference, &DrawRecords::NoteAllowed, DrawRecords::CorruptingCommits());
}

namespace {

bool compareCommitViews(const DrawRecord& record, const DrawCommitView& answer, const DrawCommitView& today, CommitDifferenceNote noteDifference, CommitAllowedNote noteAllowed, bool corrupting) {
    using DrawRecords::Field;
    if (!answer.filled) {
        noteAllowed(DrawRecords::Allowed::CommitRecordMissed);
        return false;
    }
    if (!today.filled) {
        noteAllowed(DrawRecords::Allowed::CommitTodayUnrecorded);
        return false;
    }
    const auto at = keyText(record.key);
    bool differs = false;
    const auto note = [&](Field field, const std::string& detail) {
        noteDifference(field, at + ": " + detail);
        differs = true;
    };
    if (answer.resources != today.resources) {
        if (answer.key != today.key) note(Field::Templates, "the record's template key has " + std::to_string(answer.key.size()) + " words, today's " + std::to_string(today.key.size()) + "; " + firstDifference(answer.key, today.key));
        else if (today.lookup != DrawCommitLookup::Hit) note(Field::Templates, "the record proved its template current and its moved buffers, but today's lookup " + lookupText(today, answer.resources) + " and built another");
        else noteAllowed(DrawRecords::Allowed::CommitTemplateRebuilt);
        return differs;
    }
    if (answer.pipeline != today.pipeline) note(Field::Pipelines, "the record binds another pipeline object than today's draw");
    if (answer.framebuffer != today.framebuffer) note(Field::Framebuffers, "the record renders into another framebuffer than today's draw");
    if (answer.views != today.views) note(Field::TargetViews, "the record has " + std::to_string(answer.views.size()) + " attachment views, today's draw " + std::to_string(today.views.size()));
    if (answer.targets != today.targets) note(Field::Targets, "the record's resident targets are other objects than today's");
    if (answer.passKey != today.passKey) note(Field::PassKeys, "the record's pass key " + hex(answer.passKey) + ", today's " + hex(today.passKey));
    if (answer.moved != today.moved || answer.movedSizes != today.movedSizes || answer.movedWords != today.movedWords) {
        std::size_t index = 0;
        while (index < answer.moved.size() && index < today.moved.size() && answer.moved[index] == today.moved[index] && answer.movedSizes[index] == today.movedSizes[index] && answer.movedWords[index] == today.movedWords[index]) ++index;
        const auto describe = [&](const DrawCommitView& view) { return index < view.moved.size() ? "allocation " + std::to_string(view.moved[index].first) + " at " + hex(view.moved[index].second) + " (" + std::to_string(view.movedSizes[index]) + " bytes, " + std::to_string(view.movedWords[index].size()) + " words)" : std::string("nothing"); };
        note(Field::MovedBuffers, "the record moves " + std::to_string(answer.moved.size()) + " buffers, today's draw " + std::to_string(today.moved.size()) + "; at " + std::to_string(index) + " the record has " + describe(answer) + ", today's " + describe(today));
    }
    if (answer.pushStages != today.pushStages || answer.push != today.push) {
        std::size_t dword = 0;
        while (dword * 4 < answer.push.size() && std::memcmp(answer.push.data() + dword * 4, today.push.data() + dword * 4, 4) == 0) ++dword;
        note(Field::CommitPushConstants, "the record's push constants differ at dword " + std::to_string(dword) + " (stages " + hex(answer.pushStages) + ", today's " + hex(today.pushStages) + ")");
    }
    if (!SameVertexInput(answer.vertexInput, today.vertexInput)) note(Field::VertexInputs, "the record's vertex input has " + std::to_string(answer.vertexInput.attributes.size()) + " attributes, today's " + std::to_string(today.vertexInput.attributes.size()));
    if (answer.shaderStages != today.shaderStages || answer.hasDepth != today.hasDepth) note(Field::ShaderStages, "the record's stages " + hex(answer.shaderStages) + ", today's " + hex(today.shaderStages));
    if (answer.fragmentOutputs != today.fragmentOutputs) note(Field::CommitFragmentOutputs, "the record's fragment outputs " + hex(answer.fragmentOutputs) + ", today's " + hex(today.fragmentOutputs));
    if (answer.pushed) {
        if (today.pushPipeline == nullptr) note(Field::Pipelines, "the record binds its moved buffers through push descriptors, but today's template has no push layout for this draw");
        else if (answer.pushPipeline != today.pushPipeline) note(Field::Pipelines, "the record's push pipeline is another object than today's pipeline with the push layout");
        if (today.pushPipeline != nullptr && answer.pushFramebuffer != today.pushFramebuffer) note(Field::Framebuffers, "the record's push framebuffer is another object than today's push framebuffer");
    }
    if (answer.bound && today.bound) {
        std::size_t index = 0;
        while (index < answer.bindings.size() && index < today.bindings.size() && SameBoundBuffer(answer.bindings[index], today.bindings[index])) ++index;
        if (index != answer.bindings.size() || index != today.bindings.size()) {
            const auto describe = [&](const DrawCommitView& view) { return index < view.bindings.size() ? boundText(view.bindings[index]) : std::string("nothing"); };
            note(Field::CommitBindings, "the record binds " + std::to_string(answer.bindings.size()) + std::string(answer.pushed ? " pushed" : "") + " buffer descriptors, today's draw " + std::to_string(today.bindings.size()) + "; at " + std::to_string(index) + " the record binds " + describe(answer) + ", today's " + describe(today));
        }
    }
    if (answer.access.has_value() && today.access.has_value()) {
        const auto& kept = *answer.access;
        const auto& now = *today.access;
        if (kept.readsTarget != now.readsTarget || kept.reads != now.reads || kept.writes != now.writes || kept.writeOnly != now.writeOnly || kept.images != now.images) note(Field::PassAccess, "the record's pass access memo holds " + accessText(kept) + ", today's draw derives " + accessText(now));
    }
    if (answer.keptOnce) {
        const auto* boundPipeline = answer.pushed ? today.pushPipeline : today.pipeline;
        const auto* boundFramebuffer = answer.pushed ? today.pushFramebuffer : today.framebuffer;
        if (answer.keep.resources != today.resources || answer.keep.pipeline != boundPipeline || answer.keep.framebuffer != boundFramebuffer || answer.keep.targets != today.targets) note(Field::Keeps, "the record keeps its objects once per batch, but the batch holds other objects than today's draw keeps");
        else if (today.gpuWrites != 0) note(Field::Keeps, "the record skips noting GPU writes behind its per-batch keep, but today's template notes " + std::to_string(today.gpuWrites) + " write ranges");
    }
    if (answer.passNotes) {
        for (std::size_t target = 0; target < today.notes.size(); ++target) {
            const auto& now = today.notes[target];
            if (!now.present || now.linear || !now.renote) continue;
            if (now.storesKeys || now.noteFirst != now.markedFirst || now.noteLayers != now.markedLayers) note(Field::TargetNotes, "target " + std::to_string(target) + ": the record skips its dirty note as already noted in this pass (layers " + std::to_string(now.noteFirst) + "+" + std::to_string(now.noteLayers) + "), but today's note marked layers " + std::to_string(now.markedFirst) + "+" + std::to_string(now.markedLayers) + (now.storesKeys ? " and stored the first write's DCC keys" : ""));
        }
    }
    if (corrupting && !differs) note(Field::PassKeys, "a corrupted committer answer");
    return differs;
}

}

void Driver::finishRecordDraw(const LeanDrawJob& job) {
    using DrawRecords::Counter;
    const auto& anchor = leanDraws.AnchorAt(job.anchor);
    const bool verify = (job.recordFlags & DrawRecordFlags::VerifyDue) != 0;
    std::shared_ptr<DrawRecordHit> hit;
    DrawRecordView answer;
    std::size_t checked = 0;
    if (job.records != nullptr) {
        hit = fetchDrawRecords(job, checked);
        if (hit != nullptr) {
            DrawRecords::Add(Counter::Hits);
            if (hit->inputsMoved) DrawRecords::Add(Counter::InputRefetches);
            if (hit->baseValueMoved) DrawRecords::Add(Counter::HitsBaseValue);
            if (hit->buffersPatched) DrawRecords::Add(Counter::HitsBuffersPatched);
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
            answer.fetched = std::move(hit->fetched);
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
    settleDrawRecord(job, sink, hit != nullptr, checked);
}

void Driver::verifyRecordHit(const LeanDrawJob& job, const DrawRecord& record, const DrawRecordView& answer, const DrawRecordSink& sink) {
    DrawRecords::NoteChecked(DrawRecords::Item::Fetch);
    if (!compareRecordAnswer(job, record, answer, sink, &DrawRecords::NoteDifference, &DrawRecords::NoteAllowed)) return;
    auto demote = std::make_shared<DrawRecordAnswer>();
    demote->key = record.key;
    demote->demote = true;
    DrawFinisher::AnswerLean(std::move(demote));
}

bool Driver::compareRecordAnswer(const LeanDrawJob& job, const DrawRecord& record, const DrawRecordView& answer, const DrawRecordSink& sink, void (*noteDifference)(DrawRecords::Field, std::string_view), void (*noteAllowed)(DrawRecords::Allowed)) {
    using DrawRecords::Field;
    if (!sink.queued) {
        noteAllowed(DrawRecords::Allowed::TodayNotQueued);
        return false;
    }
    const auto& today = sink.today;
    const auto at = drawText(job, record.key);
    bool differs = false;
    const auto note = [&](Field field, const std::string& detail) {
        noteDifference(field, at + ": " + detail);
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
    if (record.plan != nullptr && record.plan->addressed && answer.snapshotWords != today.snapshotWords) note(Field::Snapshots, "the record's snapshots cover " + std::to_string(answer.snapshotWords.size()) + " words, today's " + std::to_string(today.snapshotWords.size()) + "; " + firstDifference(answer.snapshotWords, today.snapshotWords));
    if (!today.snapshotWords.empty()) {
        const auto unread = [&](std::uint64_t address) {
            const auto found = std::lower_bound(today.snapshotWords.begin(), today.snapshotWords.end(), std::pair<std::uint64_t, std::uint32_t>{address, 0u});
            return found == today.snapshotWords.end() || found->first != address;
        };
        const auto first = std::find_if(answer.fetched.begin(), answer.fetched.end(), unread);
        if (first != answer.fetched.end()) note(Field::FetchedRuns, "the record fetched " + std::to_string(answer.fetched.size()) + " run words, " + std::to_string(std::count_if(first, answer.fetched.end(), unread)) + " of them at addresses today's draw did not read, the first at " + hex(*first));
    }
    if (answer.fragmentOutputs.has_value() != today.fragmentOutputs.has_value()) noteAllowed(DrawRecords::Allowed::FragmentOutputsOneSide);
    else if (answer.fragmentOutputs != today.fragmentOutputs) note(Field::FragmentOutputs, "the record's fragment outputs " + hex(*answer.fragmentOutputs) + ", today's " + hex(*today.fragmentOutputs));
    return differs;
}

void Driver::settleDrawRecord(const LeanDrawJob& job, const DrawRecordSink& sink, bool hit, std::size_t checked) {
    using DrawRecords::Counter;
    if (hit || (job.recordFlags & DrawRecordFlags::BuildWanted) == 0) return;
    if (!sink.queued || sink.contentKey.empty()) {
        DrawRecords::Add(Counter::BuildsRefused);
        return;
    }
    const auto content = sink.Content();
    const auto signature = sink.built != nullptr ? sink.built->signature : 0;
    const auto twin = job.records != nullptr ? DrawRecordTwin(*job.records, content, job.recordGeneration, job.deviceSerial, sink.built != nullptr ? &signature : nullptr) : std::nullopt;
    if (twin && *twin < checked) return;
    if (Graphics::RuntimeSwitches::Generation() != job.recordGeneration) {
        DrawRecords::Add(Counter::BuildsRefused);
        return;
    }
    auto answer = std::make_shared<DrawRecordAnswer>();
    answer->key = DrawRecordKey::For(job.shapeKey, job.parameters);
    auto built = DrawRecord::Build(answer->key, job.shape->shared_from_this(), job.deviceSerial, job.recordGeneration, Graphics::Recorder::Presents(), sink);
    if (sink.built != nullptr) built->probes = DrawRecordProbes(*sink.built, job.records);
    answer->built = std::move(built);
    DrawRecords::Add(Counter::RecordsBuilt);
    if (twin) DrawRecords::Add(Counter::RecordsRebuilt);
    DrawRecords::Add(sink.built != nullptr ? Counter::PlansBuilt : Counter::PlansRefused);
    if (!DrawFinisher::AnswerLean(std::move(answer))) {
        DrawRecords::Add(Counter::BuildsDropped);
        DrawRecords::Add(Counter::DroppedUnanswered);
    }
}

void Driver::installDrawRecord(const LeanDrawJob& job, const DrawRecordAnswer& answer) {
    using DrawRecords::Counter;
    auto& lean = leanDraws;
    const auto front = DrawFinisher::Get().Front();
    if (job.recordGeneration != lean.stamp || job.deviceSerial != lean.identitiesSerial) {
        if (answer.built != nullptr) {
            DrawRecords::Add(Counter::BuildsDropped);
            DrawRecords::Add(Counter::DroppedStale);
        }
        return;
    }
    if (answer.demote) {
        if (lean.identities.Demote(answer.key, answer.key.Probe(), front)) DrawRecords::AddOnWorker(Counter::Demotions);
        return;
    }
    if (answer.built == nullptr) return;
    const auto installed = lean.identities.Install(answer.built, Graphics::Recorder::Presents(), front);
    if (installed == DrawRecordStore::Installed::Yes) {
        DrawRecords::AddOnWorker(Counter::RecordsInstalled);
        return;
    }
    DrawRecords::Add(Counter::BuildsDropped);
    DrawRecords::Add(installed == DrawRecordStore::Installed::OverCap ? Counter::DroppedOverCap : installed == DrawRecordStore::Installed::Refused ? Counter::DroppedRefused : Counter::DroppedNoEntry);
}

void Driver::retireDrawRecord(const std::byte* payload, std::shared_ptr<const void>&& answer) {
    if (answer == nullptr) return;
    const auto& job = *std::launder(reinterpret_cast<const LeanDrawJob*>(payload));
    const auto reply = std::static_pointer_cast<const DrawRecordAnswer>(std::move(answer));
    Get().installDrawRecord(job, *reply);
}

}
