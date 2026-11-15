#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/RegisterLists.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/LeanFront.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/LeanTails.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/Submission.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::uint32_t LeanReadLimit = std::uint32_t{1} << 16u;

struct ListShape {
    std::uint64_t address = 0;
    std::uint32_t mode = 0;
    std::uint32_t count = 0;

    bool Pairs() const { return mode == RegisterListCache::PairsMode; }
    std::size_t Values() const { return Pairs() ? std::size_t{count} * 2 : count; }
    std::size_t Bytes() const { return Values() * sizeof(std::uint32_t); }
    std::size_t Words() const { return std::size_t{count} * 2; }
};

std::optional<ListShape> leanShape(std::span<const std::uint32_t> packet) {
    if (packet.size() != 5 || packet[4] > LeanReadLimit) return std::nullopt;
    return ListShape{packet[1] | (static_cast<std::uint64_t>(packet[2]) << 32u), packet[3], packet[4]};
}

bool countedReads() {
    static const bool counted = std::getenv("APS5_PROFILE_DRAW") != nullptr || std::getenv("APS5_TRACE_UNREADABLE") != nullptr;
    return counted;
}

void expand(const ListShape& list, std::uint32_t* out) {
    if (list.Pairs()) return;
    const auto* values = out + list.count;
    for (std::uint32_t index = 0; index < list.count; ++index) {
        const auto value = values[index];
        out[2 * std::size_t{index}] = list.mode + index;
        out[2 * std::size_t{index} + 1] = value;
    }
}

std::uint32_t* valuesOf(const ListShape& list, std::uint32_t* out) {
    return list.Pairs() ? out : out + list.count;
}

void copyList(const ListShape& list, std::uint32_t* out) {
    if (list.count == 0) return;
    const auto* source = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(list.address));
    GuestMemory::CheckRange(source, list.Bytes(), alignof(std::uint32_t));
    GuestMemory::CopyGuestBytes(valuesOf(list, out), source, list.Bytes());
    expand(list, out);
}

void readCounted(const ListShape& list, std::uint32_t* out) {
    if (list.count == 0) return;
    GuestMemory::Read(list.address, std::as_writable_bytes(std::span(valuesOf(list, out), list.Values())), alignof(std::uint32_t));
    expand(list, out);
}

void hookList(const ListShape& list) {
    if (list.count == 0) return;
    GuestMemory::GuardRead(list.address, list.Bytes());
    GuestMemory::FlushGpuWrites(list.address, list.Bytes());
}

std::string listError(std::uint32_t header, std::size_t cursor, const std::exception& error) {
    return "AGC driver: " + Pm4::Name(header) + " at DWORD " + std::to_string(cursor) + ": " + error.what();
}

std::string hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

std::uint64_t listVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_LEAN_LISTS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

void noteListDifference(const std::string& detail) {
    static std::atomic<std::int64_t> window{-1};
    static std::atomic<std::uint32_t> printed{0};
    const auto now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (window.exchange(now, std::memory_order_relaxed) != now) printed.store(0, std::memory_order_relaxed);
    if (printed.fetch_add(1, std::memory_order_relaxed) < 8) std::fprintf(stderr, "[lean-lists-verify] register list differs: %s\n", detail.c_str());
}

void store(LeanTails& tails, const ListShape& list, const std::uint32_t* pairs, std::uint64_t tracked, std::uint64_t epoch) {
    auto* entry = tails.cache.Find(list.address, list.mode, list.count);
    if (entry == nullptr) {
        bool evicted = false;
        entry = &tails.cache.Place(list.address, list.mode, list.count, evicted);
        entry->admitted = false;
        if (evicted) LeanFront::AddOnWorker(LeanFront::Counter::ListCacheEvictions);
    }
    entry->pairs.assign(pairs, pairs + list.Words());
    entry->tracked = tracked;
    entry->epoch = epoch;
}

void checkAnswer(const ListShape& list, std::span<const std::uint32_t> today, std::span<const std::uint32_t> lean, RegisterListCache::Entry* cached, std::uint32_t header, std::size_t cursor, std::uint64_t serial, std::uint64_t ownWrites) {
    LeanFront::NoteChecked(LeanFront::Item::RegisterList, cached != nullptr);
    if (today.size() == lean.size() && std::equal(today.begin(), today.end(), lean.begin())) return;
    std::size_t at = 0;
    while (at < today.size() && at < lean.size() && today[at] == lean[at]) ++at;
    if (cached != nullptr) {
        const auto tracked = cached->tracked;
        cached->live = false;
        if (GuestMemory::CollectWritesUncached(list.address, list.Bytes()) != 0 && !GuestMemory::UnchangedSince(list.address, list.Bytes(), tracked)) {
            LeanFront::NoteAllowed(LeanFront::Allowed::ListsRewrittenInEpoch);
            return;
        }
    }
    auto detail = "serial " + std::to_string(serial) + " " + Pm4::Name(header) + " at DWORD " + std::to_string(cursor) + ", list " + hex(list.address) + " mode " + hex(list.mode) + " count " + std::to_string(list.count) + (cached != nullptr ? " (cache hit)" : " (read)");
    if (today.size() != lean.size()) detail += ": " + std::to_string(lean.size()) + " lean words, " + std::to_string(today.size()) + " today";
    else detail += ": first difference at word " + std::to_string(at) + ": lean " + hex(lean[at]) + ", today " + hex(today[at]);
    detail += GuestMemory::OwnWriteSerial() != ownWrites ? "; driver stores landed meanwhile" : "; no driver store landed meanwhile";
    LeanFront::NoteDifference(cached != nullptr ? LeanFront::Field::ListCache : LeanFront::Field::RegisterLists, detail);
}

constexpr std::uint64_t ListSpanGap = 4096;
constexpr std::size_t NoSpan = ~std::size_t{0};

bool listSpanIdle(const std::pair<std::uint64_t, std::uint64_t>& span) {
    return Graphics::Recorder::FlushHookIdle({&span, 1}) && Graphics::ReadGuardIdle(span.first, static_cast<std::size_t>(span.second - span.first));
}

class ListCursors {
public:
    ListCursors(std::span<const std::uint32_t> words, const PacketTable* scanned) : commands(words), table(scanned) {}
    std::size_t First() {
        index = 0;
        return table == nullptr ? 0 : at();
    }
    std::size_t Next(std::size_t cursor) {
        if (table == nullptr) return cursor + Pm4::PacketWords(commands[cursor]);
        ++index;
        return at();
    }

private:
    std::size_t at() const { return index < table->lists.size() ? table->packets[table->lists[index]].cursor : commands.size(); }
    std::span<const std::uint32_t> commands;
    const PacketTable* table;
    std::size_t index = 0;
};

void listSpans(GuestRanges& spans, std::span<const std::uint32_t> commands, const PacketTable* table) {
    spans.clear();
    ListCursors walk(commands, table);
    for (std::size_t cursor = walk.First(); cursor < commands.size(); cursor = walk.Next(cursor)) {
        const auto header = commands[cursor];
        if ((header >> 30u) != 3u || !Pm4::IndirectRegisterOpcode((header >> 8u) & 0xffu)) continue;
        if (Pm4::PacketWords(header) > commands.size() - cursor) break;
        const auto shape = leanShape(commands.subspan(cursor, Pm4::PacketWords(header)));
        if (!shape || shape->count == 0 || shape->Bytes() > std::numeric_limits<std::uint64_t>::max() - shape->address) continue;
        spans.emplace_back(shape->address, shape->address + shape->Bytes());
    }
    if (spans.empty()) return;
    if (!std::is_sorted(spans.begin(), spans.end())) std::sort(spans.begin(), spans.end());
    std::size_t last = 0;
    for (std::size_t index = 1; index < spans.size(); ++index) {
        if (spans[index].first <= spans[last].second || spans[index].first - spans[last].second <= ListSpanGap) spans[last].second = std::max(spans[last].second, spans[index].second);
        else spans[++last] = spans[index];
    }
    spans.resize(last + 1);
}

void planListHooks(LeanTails& tails, std::span<const std::uint32_t> commands, const PacketTable* table) {
    auto& spans = tails.spans;
    auto& idle = tails.spanIdle;
    idle.clear();
    listSpans(spans, commands, table);
    if (spans.empty()) return;
    bool all = Graphics::Recorder::FlushHookIdle(spans);
    for (std::size_t index = 0; all && index < spans.size(); ++index) all = Graphics::ReadGuardIdle(spans[index].first, static_cast<std::size_t>(spans[index].second - spans[index].first));
    idle.assign(spans.size(), static_cast<std::uint8_t>(all ? 1 : 0));
    if (all) return;
    for (std::size_t index = 0; index < spans.size(); ++index) idle[index] = static_cast<std::uint8_t>(listSpanIdle(spans[index]) ? 1 : 0);
}

std::size_t idleListSpan(const LeanTails& tails, const ListShape& list) {
    const auto& spans = tails.spans;
    if (spans.empty() || list.count == 0) return NoSpan;
    const auto found = std::upper_bound(spans.begin(), spans.end(), list.address, [](std::uint64_t address, const std::pair<std::uint64_t, std::uint64_t>& span) { return address < span.first; });
    if (found == spans.begin()) return NoSpan;
    const auto index = static_cast<std::size_t>(found - spans.begin()) - 1;
    const auto& span = spans[index];
    if (tails.spanIdle[index] == 0 || list.address >= span.second || list.Bytes() > span.second - list.address) return NoSpan;
    return index;
}

std::uint64_t listHooksVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_LIST_HOOKS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

std::atomic<std::uint64_t> listHookTails{0};
std::atomic<std::uint64_t> listHookLists{0};
std::atomic<std::uint64_t> listHookDiffering{0};
std::atomic<std::uint64_t> listHookMoved{0};
std::atomic<std::uint64_t> listHookBytes{0};
std::atomic<std::int64_t> listHookWindow{-1};

void noteListHookCounts() {
    const std::int64_t window = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (listHookWindow.exchange(window, std::memory_order_relaxed) == window) return;
    std::fprintf(stderr, "[list-hooks-verify] %llu tails checked, %llu lists checked, %llu differing (own range not idle at its turn), %llu of them with the span no longer idle either, %llu whose bytes today's guard and hook changed\n", static_cast<unsigned long long>(listHookTails.load(std::memory_order_relaxed)), static_cast<unsigned long long>(listHookLists.load(std::memory_order_relaxed)), static_cast<unsigned long long>(listHookDiffering.load(std::memory_order_relaxed)), static_cast<unsigned long long>(listHookMoved.load(std::memory_order_relaxed)), static_cast<unsigned long long>(listHookBytes.load(std::memory_order_relaxed)));
}

const char* listHookReason(const ListShape& list) {
    const std::pair<std::uint64_t, std::uint64_t> own{list.address, list.address + list.Bytes()};
    if (Graphics::Recorder::SnapshotWriteOverlaps(list.address, list.Bytes())) return "a pending GPU write";
    if (Graphics::StorageTexture::AnyPendingOverlaps({&own, 1})) return "a pending or flushing storage image";
    if (Graphics::AnyShadowedOverlaps(list.address, list.Bytes())) return "a live unit shadow";
    if (!Graphics::ReadGuardIdle(list.address, list.Bytes())) return "the depth or multisample read guard";
    return "a queued label or the flush hook's state";
}

bool listHooksVerifyDue(LeanTails& tails) {
    const auto every = listHooksVerifyEvery();
    if (every == 0 || std::none_of(tails.spanIdle.begin(), tails.spanIdle.end(), [](std::uint8_t idle) { return idle != 0; })) return false;
    if (++tails.hookVerifyOrdinal % every != 0) return false;
    listHookTails.fetch_add(1, std::memory_order_relaxed);
    noteListHookCounts();
    return true;
}

void checkListHook(const LeanTails& tails, const ListShape& list, std::size_t span, std::uint64_t serial, std::size_t cursor) {
    listHookLists.fetch_add(1, std::memory_order_relaxed);
    const std::pair<std::uint64_t, std::uint64_t> own{list.address, list.address + list.Bytes()};
    if (!Graphics::Recorder::FlushHookIdle({&own, 1}) || !Graphics::ReadGuardIdle(list.address, list.Bytes())) {
        const auto& range = tails.spans[span];
        const bool spanIdle = listSpanIdle(range);
        if (!spanIdle) listHookMoved.fetch_add(1, std::memory_order_relaxed);
        if (listHookDiffering.fetch_add(1, std::memory_order_relaxed) < 16) {
            std::fprintf(stderr, "[list-hooks-verify] serial %llu list at DWORD %zu, 0x%llx+0x%zx: its guard and hook were skipped, its own range is not idle at its turn (%s); its span 0x%llx..0x%llx %s\n", static_cast<unsigned long long>(serial), cursor, static_cast<unsigned long long>(list.address), list.Bytes(), listHookReason(list), static_cast<unsigned long long>(range.first), static_cast<unsigned long long>(range.second), spanIdle ? "still answers idle" : "no longer answers idle either");
        }
    }
    const auto* source = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(list.address));
    const bool readable = GuestMemory::Accessible(source, list.Bytes());
    std::vector<std::uint8_t> before;
    if (readable) {
        before.resize(list.Bytes());
        GuestMemory::CopyGuestBytes(before.data(), source, before.size());
    }
    hookList(list);
    if (!readable) return;
    std::vector<std::uint8_t> after(before.size());
    GuestMemory::CopyGuestBytes(after.data(), source, after.size());
    const auto differs = std::mismatch(before.begin(), before.end(), after.begin());
    if (differs.first == before.end()) return;
    if (listHookBytes.fetch_add(1, std::memory_order_relaxed) < 16) {
        std::fprintf(stderr, "[list-hooks-verify] serial %llu list at DWORD %zu, 0x%llx+0x%zx: today's guard and hook changed its bytes, the first at offset 0x%zx\n", static_cast<unsigned long long>(serial), cursor, static_cast<unsigned long long>(list.address), list.Bytes(), static_cast<std::size_t>(differs.first - before.begin()));
    }
}

}

const RegisterListRef* RegisterLists::Find(std::size_t cursor) const {
    const auto found = std::lower_bound(refs.begin(), refs.end(), cursor, [](const RegisterListRef& ref, std::size_t at) { return ref.cursor < at; });
    return found != refs.end() && found->cursor == cursor ? &*found : nullptr;
}

const RegisterListRef* RegisterLists::Next(std::size_t cursor, std::size_t& hint) const {
    if (hint > refs.size() || (hint > 0 && refs[hint - 1].cursor >= cursor)) {
        hint = static_cast<std::size_t>(std::lower_bound(refs.begin(), refs.end(), cursor, [](const RegisterListRef& ref, std::size_t at) { return ref.cursor < at; }) - refs.begin());
    }
    while (hint < refs.size() && refs[hint].cursor < cursor) ++hint;
    return hint < refs.size() && refs[hint].cursor == cursor ? &refs[hint] : nullptr;
}

RegisterListView RegisterListAt(const Submission& submission, std::size_t cursor, std::size_t& hint) {
    if (!submission.leanLists) {
        const auto& listed = submission.registerLists.at(cursor);
        return {listed, Pm4::IndirectRegistersValid(listed)};
    }
    const auto* ref = submission.lists.Next(cursor, hint);
    if (ref == nullptr) throw std::out_of_range("AGC driver: no register list was read for the packet at DWORD " + std::to_string(cursor));
    return submission.lists.View(*ref);
}

std::optional<RegisterListView> FindRegisterList(const Submission& submission, std::size_t cursor) {
    if (!submission.leanLists) {
        const auto found = submission.registerLists.find(cursor);
        if (found == submission.registerLists.end()) return std::nullopt;
        return RegisterListView{found->second, Pm4::IndirectRegistersValid(found->second)};
    }
    const auto* ref = submission.lists.Find(cursor);
    if (ref == nullptr) return std::nullopt;
    return submission.lists.View(*ref);
}

std::uint64_t RegisterListCache::mix(std::uint64_t address, std::uint32_t mode, std::uint32_t count) {
    const auto mixed = (address >> 2u) ^ (static_cast<std::uint64_t>(mode) << 21u) ^ (static_cast<std::uint64_t>(count) << 47u);
    return mixed * 0x9e3779b97f4a7c15ull;
}

std::size_t RegisterListCache::setOf(std::uint64_t address, std::uint32_t mode, std::uint32_t count) const {
    return static_cast<std::size_t>(mix(address, mode, count) >> shift) & (sets - 1);
}

RegisterListCache::Entry* RegisterListCache::Find(std::uint64_t address, std::uint32_t mode, std::uint32_t count) {
    if (entries.empty()) return nullptr;
    auto* set = entries.data() + setOf(address, mode, count) * ways;
    for (std::size_t way = 0; way < ways; ++way) {
        if (set[way].Is(address, mode, count)) {
            set[way].used = clock;
            return &set[way];
        }
    }
    return nullptr;
}

RegisterListCache::Entry& RegisterListCache::Place(std::uint64_t address, std::uint32_t mode, std::uint32_t count, bool& evicted) {
    if (entries.empty()) entries.resize(sets * ways);
    auto* set = entries.data() + setOf(address, mode, count) * ways;
    Entry* chosen = nullptr;
    for (std::size_t way = 0; way < ways && chosen == nullptr; ++way) {
        if (set[way].Is(address, mode, count)) chosen = &set[way];
    }
    for (std::size_t way = 0; way < ways && chosen == nullptr; ++way) {
        if (!set[way].live) chosen = &set[way];
    }
    evicted = chosen == nullptr;
    if (chosen == nullptr) {
        chosen = set;
        for (std::size_t way = 1; way < ways; ++way) {
            if (set[way].used < chosen->used) chosen = &set[way];
        }
        if (wide) {
            const auto tag = mix(chosen->address, chosen->mode, chosen->count);
            ghosts[static_cast<std::size_t>(tag >> 50u) % GhostSlots] = tag | 1u;
        }
    }
    chosen->address = address;
    chosen->mode = mode;
    chosen->count = count;
    chosen->used = clock;
    chosen->live = true;
    return *chosen;
}

bool RegisterListCache::Volatile(std::uint64_t address) const {
    if (blocks.empty()) return false;
    const auto block = address >> 16u;
    const auto& slot = blocks[static_cast<std::size_t>((block * 0x9e3779b97f4a7c15ull) >> 52u) % BlockSlots];
    return slot.block == block && slot.declines >= VolatileDeclines;
}

void RegisterListCache::Decline(std::uint64_t address, std::uint32_t weight) {
    if (blocks.empty()) blocks.resize(BlockSlots);
    const auto block = address >> 16u;
    auto& slot = blocks[static_cast<std::size_t>((block * 0x9e3779b97f4a7c15ull) >> 52u) % BlockSlots];
    if (slot.block != block) slot = Block{block, 0};
    slot.declines = std::min(VolatileDeclines, slot.declines + weight);
}

void RegisterListCache::Tick() {
    ++clock;
    if (++declinesSince < DeclineWindow) return;
    declinesSince = 0;
    std::fill(blocks.begin(), blocks.end(), Block{});
}

void RegisterListCache::Clear() {
    for (auto& entry : entries) {
        entry.live = false;
        entry.admitted = false;
    }
    std::fill(blocks.begin(), blocks.end(), Block{});
    declinesSince = 0;
    if (!wide) return;
    wide = false;
    sets = Sets;
    ways = Ways;
    shift = 53;
    entries.clear();
    std::vector<std::uint64_t>().swap(ghosts);
    refused.clear();
}

void RegisterListCache::Widen() {
    if (wide) return;
    wide = true;
    sets = WideSets;
    ways = WideWays;
    shift = 52;
    entries.clear();
    ghosts.assign(GhostSlots, 0);
    refused.clear();
}

bool RegisterListCache::Evicted(std::uint64_t address, std::uint32_t mode, std::uint32_t count) const {
    if (!wide) return false;
    const auto tag = mix(address, mode, count);
    return ghosts[static_cast<std::size_t>(tag >> 50u) % GhostSlots] == (tag | 1u);
}

void RegisterListCache::Refuse(std::uint64_t address) {
    if (!Refused(address)) refused.push_back(address);
}

void Driver::resolveLeanLists(Submission& submission, bool cacheable, const PacketTable* table) {
    if (cacheable && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::LeanListDirect)) {
        cacheable = false;
        LeanFront::AddOnWorker(LeanFront::Counter::DirectListTails);
    }
    if (cacheable && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::LeanListBatch) && !countedReads()) {
        resolveBatchedLists(submission);
        return;
    }
    using LeanFront::Counter;
    auto& lists = submission.lists;
    lists = leanTails.TakeLists();
    submission.leanLists = true;
    auto& cache = leanTails.cache;
    auto& hits = leanTails.hits;
    hits.clear();
    if (cacheable) cache.Tick();
    const auto epoch = cacheable ? GuestMemory::ThreadCollectEpoch() : 0;
    const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::Registers);
    const auto& commands = submission.commands;
    std::uint64_t resolved = 0;
    const bool hooksOnce = LeanFront::SwitchSet(frontLatch, Graphics::RuntimeSwitch::ListHooksOnce);
    leanTails.spans.clear();
    if (hooksOnce) planListHooks(leanTails, commands, table);
    const bool verifyHooks = hooksOnce && listHooksVerifyDue(leanTails);
    const auto ownRead = [&] { leanTails.spans.clear(); };
    const auto hook = [&](const ListShape& list, std::size_t at) {
        if (const auto span = idleListSpan(leanTails, list); span != NoSpan) {
            if (verifyHooks) checkListHook(leanTails, list, span, submission.serial, at);
            return;
        }
        ownRead();
        hookList(list);
    };
    ListCursors walk(commands, table);
    for (std::size_t cursor = walk.First(); cursor < commands.size(); cursor = walk.Next(cursor)) {
        const auto header = commands[cursor];
        if ((header >> 30u) != 3u || !Pm4::IndirectRegisterOpcode((header >> 8u) & 0xffu)) continue;
        const auto packet = std::span<const std::uint32_t>(commands).subspan(cursor, Pm4::PacketWords(header));
        ++resolved;
        try {
            const auto first = lists.words.size();
            const auto shape = leanShape(packet);
            if (!shape) {
                ownRead();
                const auto pairs = Pm4::ReadIndirectRegisters(packet);
                lists.words.insert(lists.words.end(), pairs.begin(), pairs.end());
                lists.refs.push_back({static_cast<std::uint32_t>(cursor), static_cast<std::uint32_t>(first), static_cast<std::uint32_t>(pairs.size() / 2), Pm4::IndirectRegistersValid(pairs) ? RegisterLists::Valid : 0u});
                continue;
            }
            const bool candidate = cacheable && RegisterListCache::Cacheable(shape->count);
            const bool bypass = candidate && cache.Volatile(shape->address);
            auto* entry = candidate && !bypass ? cache.Find(shape->address, shape->mode, shape->count) : nullptr;
            if (LeanFront::Due(frontLatch, LeanFront::Item::RegisterList)) {
                ownRead();
                const auto ownWrites = GuestMemory::OwnWriteSerial();
                const auto tracked = candidate && !bypass ? GuestMemory::CollectWrites(shape->address, shape->Bytes()) : 0;
                const auto pairs = Pm4::ReadIndirectRegisters(packet);
                const bool valid = Pm4::IndirectRegistersValid(pairs);
                lists.words.insert(lists.words.end(), pairs.begin(), pairs.end());
                lists.refs.push_back({static_cast<std::uint32_t>(cursor), static_cast<std::uint32_t>(first), shape->count, valid ? RegisterLists::Valid : 0u});
                auto& lean = leanTails.scratch;
                lean.assign(shape->Words(), 0);
                RegisterListCache::Entry* cached = nullptr;
                if (entry != nullptr && tracked == 0) entry->live = false;
                if (entry != nullptr && tracked != 0) {
                    entry->epoch = epoch;
                    if (GuestMemory::UnchangedSince(shape->address, shape->Bytes(), entry->tracked)) {
                        std::copy(entry->pairs.begin(), entry->pairs.end(), lean.begin());
                        cached = entry;
                    }
                }
                bool answered = true;
                try {
                    if (cached == nullptr) copyList(*shape, lean.data());
                } catch (const std::exception& error) {
                    answered = false;
                    LeanFront::NoteChecked(LeanFront::Item::RegisterList);
                    LeanFront::NoteDifference(LeanFront::Field::RegisterLists, "serial " + std::to_string(submission.serial) + " " + listError(header, cursor, error) + " in the lean read after today's read succeeded");
                }
                if (answered) checkAnswer(*shape, pairs, lean, cached, header, cursor, submission.serial, ownWrites);
                if (tracked != 0 && valid) store(leanTails, *shape, pairs.data(), tracked, epoch);
                else if (candidate && !bypass && tracked == 0) cache.Decline(shape->address, RegisterListCache::VolatileDeclines);
                continue;
            }
            lists.words.resize(first + shape->Words());
            auto* out = lists.words.data() + first;
            std::uint32_t flags = 0;
            if (shape->count == 0) {
                flags = RegisterLists::Valid;
            } else if (entry != nullptr) {
                hook(*shape, cursor);
                if (entry->epoch != epoch) {
                    if (GuestMemory::CollectWrites(shape->address, shape->Bytes()) == 0) entry->live = false;
                    else entry->epoch = epoch;
                }
                if (entry->live) {
                    std::copy(entry->pairs.begin(), entry->pairs.end(), out);
                    hits.push_back({static_cast<std::uint32_t>(lists.refs.size()), static_cast<std::uint32_t>(cursor), shape->address, shape->mode, shape->count, entry->tracked, entry});
                    flags = RegisterLists::Valid | RegisterLists::Cached;
                } else {
                    cache.Decline(shape->address, RegisterListCache::VolatileDeclines);
                    copyList(*shape, out);
                    flags = Pm4::IndirectRegistersValid(std::span<const std::uint32_t>(out, shape->Words())) ? RegisterLists::Valid : 0u;
                    LeanFront::AddOnWorker(Counter::ListCacheMisses);
                }
            } else {
                std::uint64_t tracked = 0;
                if (countedReads()) {
                    ownRead();
                    if (candidate && !bypass) tracked = GuestMemory::CollectWrites(shape->address, shape->Bytes());
                    readCounted(*shape, out);
                } else {
                    hook(*shape, cursor);
                    if (candidate && !bypass) tracked = GuestMemory::CollectWrites(shape->address, shape->Bytes());
                    copyList(*shape, out);
                }
                const bool valid = Pm4::IndirectRegistersValid(std::span<const std::uint32_t>(out, shape->Words()));
                flags = valid ? RegisterLists::Valid : 0u;
                if (bypass) {
                    LeanFront::AddOnWorker(Counter::ListCacheBypassed);
                } else if (candidate) {
                    LeanFront::AddOnWorker(Counter::ListCacheMisses);
                    if (tracked != 0 && valid) store(leanTails, *shape, out, tracked, epoch);
                    else if (tracked == 0) cache.Decline(shape->address, RegisterListCache::VolatileDeclines);
                }
            }
            lists.refs.push_back({static_cast<std::uint32_t>(cursor), static_cast<std::uint32_t>(first), shape->count, flags});
        } catch (const std::exception& error) {
            throw std::runtime_error(listError(header, cursor, error));
        }
    }
    LeanFront::AddOnWorker(Counter::LeanLists, resolved);
    if (!hits.empty()) settleListHits(submission);
}

void Driver::resolveBatchedLists(Submission& submission) {
    using LeanFront::Counter;
    auto& lists = submission.lists;
    lists = leanTails.TakeLists();
    submission.leanLists = true;
    auto& cache = leanTails.cache;
    cache.Widen();
    auto& hits = leanTails.hits;
    hits.clear();
    cache.Tick();
    const auto epoch = GuestMemory::ThreadCollectEpoch();
    const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::Registers);
    const auto& commands = submission.commands;
    auto& plan = leanTails.plan;
    auto& collects = leanTails.collects;
    auto& servedBatched = leanTails.servedBatched;
    plan.clear();
    collects.clear();
    servedBatched.clear();
    for (std::size_t cursor = 0; cursor < commands.size(); cursor += Pm4::PacketWords(commands[cursor])) {
        const auto header = commands[cursor];
        if ((header >> 30u) != 3u || !Pm4::IndirectRegisterOpcode((header >> 8u) & 0xffu)) continue;
        const auto packet = std::span<const std::uint32_t>(commands).subspan(cursor, Pm4::PacketWords(header));
        BatchedList item;
        item.cursor = static_cast<std::uint32_t>(cursor);
        if (const auto shape = leanShape(packet)) {
            item.lean = true;
            item.due = LeanFront::Due(frontLatch, LeanFront::Item::RegisterList);
            if (!item.due && shape->count != 0) {
                try {
                    hookList(*shape);
                } catch (const std::exception& error) {
                    throw std::runtime_error(listError(header, cursor, error));
                }
                const bool candidate = RegisterListCache::Cacheable(shape->count);
                item.looked = true;
                item.bypass = candidate && (cache.Refused(shape->address) || cache.Volatile(shape->address));
                item.entry = candidate && !item.bypass ? cache.Find(shape->address, shape->mode, shape->count) : nullptr;
                if (item.entry != nullptr && item.entry->epoch != epoch) {
                    collects.push_back({shape->address, shape->Bytes(), 0});
                    item.query = static_cast<std::uint32_t>(collects.size());
                }
            }
        }
        plan.push_back(item);
    }
    if (!collects.empty()) {
        const auto counts = GuestMemory::CollectWritesBatch(collects);
        LeanFront::AddOnWorker(Counter::ListBatchQueries);
        LeanFront::AddOnWorker(Counter::ListBatchRanges, collects.size());
        LeanFront::AddOnWorker(Counter::ListBatchMemo, counts.memo);
        LeanFront::AddOnWorker(Counter::ListBatchSpans, counts.spans);
        LeanFront::AddOnWorker(Counter::ListBatchBytes, counts.bytes);
        LeanFront::AddOnWorker(Counter::ListBatchAlone, counts.alone);
    }
    std::uint64_t resolved = 0;
    std::uint64_t mutations = 0;
    for (const auto& item : plan) {
        const std::size_t cursor = item.cursor;
        const auto header = commands[cursor];
        const auto packet = std::span<const std::uint32_t>(commands).subspan(cursor, Pm4::PacketWords(header));
        ++resolved;
        try {
            const auto first = lists.words.size();
            if (!item.lean) {
                const auto pairs = Pm4::ReadIndirectRegisters(packet);
                lists.words.insert(lists.words.end(), pairs.begin(), pairs.end());
                lists.refs.push_back({static_cast<std::uint32_t>(cursor), static_cast<std::uint32_t>(first), static_cast<std::uint32_t>(pairs.size() / 2), Pm4::IndirectRegistersValid(pairs) ? RegisterLists::Valid : 0u});
                continue;
            }
            const auto shape = *leanShape(packet);
            const bool candidate = RegisterListCache::Cacheable(shape.count);
            const bool carried = item.looked && mutations == 0;
            const bool bypass = carried ? item.bypass : candidate && (cache.Refused(shape.address) || cache.Volatile(shape.address));
            auto* entry = carried ? item.entry : candidate && !bypass ? cache.Find(shape.address, shape.mode, shape.count) : nullptr;
            if (item.due) {
                ++mutations;
                const auto ownWrites = GuestMemory::OwnWriteSerial();
                const auto tracked = candidate && !bypass ? GuestMemory::CollectWrites(shape.address, shape.Bytes()) : 0;
                const auto pairs = Pm4::ReadIndirectRegisters(packet);
                const bool valid = Pm4::IndirectRegistersValid(pairs);
                lists.words.insert(lists.words.end(), pairs.begin(), pairs.end());
                lists.refs.push_back({static_cast<std::uint32_t>(cursor), static_cast<std::uint32_t>(first), shape.count, valid ? RegisterLists::Valid : 0u});
                auto& lean = leanTails.scratch;
                lean.assign(shape.Words(), 0);
                RegisterListCache::Entry* cached = nullptr;
                if (entry != nullptr && tracked == 0) entry->live = false;
                if (entry != nullptr && tracked != 0) {
                    entry->epoch = epoch;
                    if (GuestMemory::UnchangedSince(shape.address, shape.Bytes(), entry->tracked)) {
                        std::copy(entry->pairs.begin(), entry->pairs.end(), lean.begin());
                        cached = entry;
                    }
                }
                bool answered = true;
                try {
                    if (cached == nullptr) copyList(shape, lean.data());
                } catch (const std::exception& error) {
                    answered = false;
                    LeanFront::NoteChecked(LeanFront::Item::RegisterList);
                    LeanFront::NoteDifference(LeanFront::Field::RegisterLists, "serial " + std::to_string(submission.serial) + " " + listError(header, cursor, error) + " in the lean read after today's read succeeded");
                }
                if (answered) checkAnswer(shape, pairs, lean, cached, header, cursor, submission.serial, ownWrites);
                if (tracked != 0 && valid) store(leanTails, shape, pairs.data(), tracked, epoch);
                else if (candidate && !bypass && tracked == 0) cache.Decline(shape.address, RegisterListCache::VolatileDeclines);
                continue;
            }
            lists.words.resize(first + shape.Words());
            auto* out = lists.words.data() + first;
            std::uint32_t flags = 0;
            if (shape.count == 0) {
                flags = RegisterLists::Valid;
            } else if (entry != nullptr) {
                bool batched = false;
                if (entry->epoch != epoch) {
                    std::uint64_t collected = 0;
                    if (item.query != 0) {
                        collected = collects[item.query - 1].generation;
                        batched = true;
                    } else {
                        collected = GuestMemory::CollectWrites(shape.address, shape.Bytes());
                    }
                    if (collected == 0) entry->live = false;
                    else entry->epoch = epoch;
                }
                if (entry->live) {
                    std::copy(entry->pairs.begin(), entry->pairs.end(), out);
                    hits.push_back({static_cast<std::uint32_t>(lists.refs.size()), static_cast<std::uint32_t>(cursor), shape.address, shape.mode, shape.count, entry->tracked, entry});
                    servedBatched.push_back(batched ? 1 : 0);
                    flags = RegisterLists::Valid | RegisterLists::Cached;
                } else {
                    ++mutations;
                    cache.Decline(shape.address, RegisterListCache::VolatileDeclines);
                    copyList(shape, out);
                    flags = Pm4::IndirectRegistersValid(std::span<const std::uint32_t>(out, shape.Words())) ? RegisterLists::Valid : 0u;
                    LeanFront::AddOnWorker(Counter::ListCacheMisses);
                }
            } else {
                std::uint64_t tracked = 0;
                if (candidate && !bypass) tracked = GuestMemory::CollectWrites(shape.address, shape.Bytes());
                copyList(shape, out);
                const bool valid = Pm4::IndirectRegistersValid(std::span<const std::uint32_t>(out, shape.Words()));
                flags = valid ? RegisterLists::Valid : 0u;
                if (bypass) {
                    LeanFront::AddOnWorker(Counter::ListCacheBypassed);
                } else if (candidate) {
                    ++mutations;
                    LeanFront::AddOnWorker(Counter::ListCacheMisses);
                    if (cache.Evicted(shape.address, shape.mode, shape.count)) LeanFront::AddOnWorker(Counter::ListMissesEvicted);
                    if (tracked != 0 && valid) store(leanTails, shape, out, tracked, epoch);
                    else if (tracked == 0) cache.Decline(shape.address, RegisterListCache::VolatileDeclines);
                }
            }
            lists.refs.push_back({static_cast<std::uint32_t>(cursor), static_cast<std::uint32_t>(first), shape.count, flags});
        } catch (const std::exception& error) {
            throw std::runtime_error(listError(header, cursor, error));
        }
    }
    LeanFront::AddOnWorker(Counter::LeanLists, resolved);
    if (hits.empty()) return;
    leanTails.served.assign(hits.begin(), hits.end());
    settleListHits(submission);
    countServedLists(submission);
}

void Driver::countServedLists(const Submission& submission) {
    using LeanFront::Counter;
    const auto every = listVerifyEvery();
    auto& cache = leanTails.cache;
    const auto& lists = submission.lists;
    const auto& served = leanTails.served;
    for (std::size_t index = 0; index < served.size(); ++index) {
        const auto& hit = served[index];
        const bool batched = leanTails.servedBatched[index] != 0;
        const auto& ref = lists.refs[hit.ref];
        if ((ref.flags & RegisterLists::Cached) == 0) continue;
        LeanFront::AddOnWorker(batched ? Counter::ListBatchHits : Counter::ListWalkHits);
        if (every == 0 || ++leanTails.verifyOrdinal % every != 0) continue;
        const ListShape shape{hit.address, hit.mode, hit.count};
        auto& fresh = leanTails.scratch;
        fresh.assign(shape.Words(), 0);
        const auto where = [&] { return "serial " + std::to_string(submission.serial) + " " + Pm4::Name(submission.commands[hit.cursor]) + " at DWORD " + std::to_string(hit.cursor) + ", list " + hex(hit.address) + " mode " + hex(hit.mode) + " count " + std::to_string(hit.count) + (batched ? " (batched query)" : " (today's walk)"); };
        LeanFront::AddOnWorker(Counter::ListVerifyChecked);
        try {
            copyList(shape, fresh.data());
        } catch (const std::exception& error) {
            LeanFront::AddOnWorker(Counter::ListVerifyDiffering);
            cache.Refuse(hit.address);
            noteListDifference(where() + ": today's read failed: " + error.what());
            continue;
        }
        const auto answer = std::span<const std::uint32_t>(lists.words).subspan(ref.first, shape.Words());
        const bool valid = Pm4::IndirectRegistersValid(fresh);
        if (std::equal(answer.begin(), answer.end(), fresh.begin()) && valid == ((ref.flags & RegisterLists::Valid) != 0)) continue;
        if (auto* entry = cache.Find(hit.address, hit.mode, hit.count); entry != nullptr && entry->tracked == hit.tracked) entry->live = false;
        if (GuestMemory::CollectWritesUncached(hit.address, shape.Bytes()) != 0 && !GuestMemory::UnchangedSince(hit.address, shape.Bytes(), hit.tracked)) {
            LeanFront::AddOnWorker(Counter::ListVerifyRewritten);
            continue;
        }
        LeanFront::AddOnWorker(Counter::ListVerifyDiffering);
        cache.Refuse(hit.address);
        std::size_t at = 0;
        while (at < answer.size() && answer[at] == fresh[at]) ++at;
        auto detail = where();
        if (at < answer.size()) detail += ": first difference at word " + std::to_string(at) + ": lean " + hex(answer[at]) + ", today " + hex(fresh[at]);
        else detail += std::string(": the lean flag says ") + (valid ? "invalid" : "valid") + ", today's read is " + (valid ? "valid" : "invalid");
        noteListDifference(detail);
    }
    leanTails.served.clear();
}

void Driver::settleListHits(Submission& submission) {
    using LeanFront::Counter;
    auto& hits = leanTails.hits;
    auto& queries = leanTails.queries;
    queries.clear();
    for (const auto& hit : hits) queries.push_back({hit.address, ListShape{hit.address, hit.mode, hit.count}.Bytes(), hit.tracked});
    const bool all = GuestMemory::UnchangedSinceAll(queries);
    const auto epoch = GuestMemory::ThreadCollectEpoch();
    auto& lists = submission.lists;
    for (const auto& hit : hits) {
        const ListShape shape{hit.address, hit.mode, hit.count};
        auto* entry = hit.entry != nullptr && hit.entry->Is(hit.address, hit.mode, hit.count) && hit.entry->tracked == hit.tracked ? hit.entry : nullptr;
        if (all || GuestMemory::UnchangedSince(hit.address, shape.Bytes(), hit.tracked)) {
            LeanFront::AddOnWorker(Counter::ListCacheHits);
            if (entry != nullptr && !entry->admitted) {
                entry->admitted = true;
                LeanFront::AddOnWorker(Counter::ListCacheAdmits);
            }
            continue;
        }
        LeanFront::AddOnWorker(Counter::ListCacheInvalidations);
        leanTails.cache.Decline(hit.address);
        auto& ref = lists.refs[hit.ref];
        auto* out = lists.words.data() + ref.first;
        std::uint64_t tracked = 0;
        try {
            tracked = GuestMemory::CollectWrites(hit.address, shape.Bytes());
            copyList(shape, out);
        } catch (const std::exception& error) {
            throw std::runtime_error(listError(submission.commands[hit.cursor], hit.cursor, error));
        }
        const bool valid = Pm4::IndirectRegistersValid(std::span<const std::uint32_t>(out, shape.Words()));
        ref.flags = valid ? RegisterLists::Valid : 0u;
        if (entry == nullptr) continue;
        if (tracked == 0 || !valid) {
            entry->live = false;
            continue;
        }
        entry->pairs.assign(out, out + shape.Words());
        entry->tracked = tracked;
        entry->epoch = epoch;
        entry->admitted = false;
    }
    hits.clear();
}

void Driver::verifyLeanLists(const Submission& submission, bool cacheable, std::uint64_t ownWrites) {
    auto& cache = leanTails.cache;
    const auto epoch = cacheable ? GuestMemory::ThreadCollectEpoch() : 0;
    const auto& commands = submission.commands;
    for (std::size_t cursor = 0; cursor < commands.size(); cursor += Pm4::PacketWords(commands[cursor])) {
        const auto header = commands[cursor];
        if ((header >> 30u) != 3u || !Pm4::IndirectRegisterOpcode((header >> 8u) & 0xffu)) continue;
        const auto packet = std::span<const std::uint32_t>(commands).subspan(cursor, Pm4::PacketWords(header));
        const auto shape = leanShape(packet);
        const auto found = submission.registerLists.find(cursor);
        if (!shape || found == submission.registerLists.end()) continue;
        const bool candidate = cacheable && RegisterListCache::Cacheable(shape->count);
        auto* entry = candidate && !cache.Volatile(shape->address) ? cache.Find(shape->address, shape->mode, shape->count) : nullptr;
        auto& lean = leanTails.scratch;
        lean.assign(shape->Words(), 0);
        RegisterListCache::Entry* cached = nullptr;
        try {
            if (entry != nullptr && (entry->epoch == epoch || GuestMemory::CollectWrites(shape->address, shape->Bytes()) != 0)) {
                entry->epoch = epoch;
                if (GuestMemory::UnchangedSince(shape->address, shape->Bytes(), entry->tracked)) {
                    std::copy(entry->pairs.begin(), entry->pairs.end(), lean.begin());
                    cached = entry;
                }
            }
            if (cached == nullptr) copyList(*shape, lean.data());
        } catch (const std::exception& error) {
            LeanFront::NoteDifference(LeanFront::Field::RegisterLists, "serial " + std::to_string(submission.serial) + " " + listError(header, cursor, error) + " in the lean read after today's read succeeded");
            continue;
        }
        checkAnswer(*shape, found->second, lean, cached, header, cursor, submission.serial, ownWrites);
    }
}

namespace {

constexpr std::uint64_t AheadPageBytes = 4096;

void mergeSpans(GuestRanges& spans, std::uint64_t gap) {
    if (spans.empty()) return;
    std::sort(spans.begin(), spans.end());
    std::size_t last = 0;
    for (std::size_t index = 1; index < spans.size(); ++index) {
        if (spans[index].first <= spans[last].second || spans[index].first - spans[last].second <= gap) spans[last].second = std::max(spans[last].second, spans[index].second);
        else spans[++last] = spans[index];
    }
    spans.resize(last + 1);
}

std::string aheadListWhere(std::span<const std::uint32_t> commands, std::size_t cursor) {
    return Pm4::Name(commands[cursor]) + " at DWORD " + std::to_string(cursor);
}

}

AheadTail::Lists ResolveListsAhead(AheadTail& job) {
    using Lists = AheadTail::Lists;
    auto& lists = job.resolved;
    lists.Clear();
    job.collectSpans.clear();
    job.queries.clear();
    job.listPackets = 0;
    const std::span<const std::uint32_t> commands = job.copy.commands;
    const PacketTable* table = job.byTable ? &job.table : nullptr;
    listSpans(job.hookSpans, commands, table);
    GuestRanges exact;
    ListCursors walk(commands, table);
    for (std::size_t cursor = walk.First(); cursor < commands.size(); cursor = walk.Next(cursor)) {
        const auto header = commands[cursor];
        if ((header >> 30u) != 3u || !Pm4::IndirectRegisterOpcode((header >> 8u) & 0xffu)) continue;
        if (Pm4::PacketWords(header) > commands.size() - cursor) return Lists::Shape;
        ++job.listPackets;
        const auto shape = leanShape(commands.subspan(cursor, Pm4::PacketWords(header)));
        if (!shape) return Lists::Shape;
        if (shape->count == 0) continue;
        if (shape->address > std::numeric_limits<std::uint64_t>::max() - AheadPageBytes || shape->Bytes() > std::numeric_limits<std::uint64_t>::max() - AheadPageBytes - shape->address) return Lists::Unreadable;
        const auto end = shape->address + shape->Bytes();
        exact.emplace_back(shape->address, end);
        job.collectSpans.emplace_back(shape->address & ~(AheadPageBytes - 1), (end + AheadPageBytes - 1) & ~(AheadPageBytes - 1));
    }
    mergeSpans(exact, 0);
    mergeSpans(job.collectSpans, 0);
    const auto abandoned = [&] { return job.state.load(std::memory_order_acquire) == AheadTail::Abandoned; };
    if (abandoned()) return Lists::Dropped;
    job.listForgets = GuestMemory::ForgetSerial();
    if ((job.listForgets & 1u) != 0) return Lists::Forgetting;
    std::uint64_t generation = 0;
    for (const auto& [begin, end] : job.collectSpans) {
        const auto collected = GuestMemory::CollectWrites(begin, static_cast<std::size_t>(end - begin));
        if (collected == 0) return Lists::Unwatched;
        generation = std::max(generation, collected);
    }
    for (const auto& [begin, end] : exact) job.queries.push_back({begin, static_cast<std::size_t>(end - begin), generation});
    const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::Registers);
    for (std::size_t cursor = walk.First(); cursor < commands.size(); cursor = walk.Next(cursor)) {
        const auto header = commands[cursor];
        if ((header >> 30u) != 3u || !Pm4::IndirectRegisterOpcode((header >> 8u) & 0xffu)) continue;
        const auto shape = leanShape(commands.subspan(cursor, Pm4::PacketWords(header)));
        if (!shape) return Lists::Shape;
        const auto first = lists.words.size();
        if (shape->count == 0) {
            lists.refs.push_back({static_cast<std::uint32_t>(cursor), static_cast<std::uint32_t>(first), 0u, RegisterLists::Valid});
            continue;
        }
        if (abandoned()) return Lists::Dropped;
        lists.words.resize(first + shape->Words());
        auto* out = lists.words.data() + first;
        try {
            copyList(*shape, out);
        } catch (const std::exception&) {
            return Lists::Unreadable;
        }
        const bool valid = Pm4::IndirectRegistersValid(std::span<const std::uint32_t>(out, shape->Words()));
        lists.refs.push_back({static_cast<std::uint32_t>(cursor), static_cast<std::uint32_t>(first), shape->count, valid ? RegisterLists::Valid : 0u});
    }
    return Lists::Built;
}

bool CheckAheadLists(const Submission& submission, const PacketTable* table, std::string& detail) {
    const std::span<const std::uint32_t> commands = submission.commands;
    const auto& lists = submission.lists;
    std::vector<std::uint32_t> fresh;
    std::size_t index = 0;
    std::size_t first = 0;
    const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::Registers);
    ListCursors walk(commands, table);
    for (std::size_t cursor = walk.First(); cursor < commands.size(); cursor = walk.Next(cursor)) {
        const auto header = commands[cursor];
        if ((header >> 30u) != 3u || !Pm4::IndirectRegisterOpcode((header >> 8u) & 0xffu)) continue;
        if (Pm4::PacketWords(header) > commands.size() - cursor) {
            detail = aheadListWhere(commands, cursor) + ": today's walk finds a truncated list packet";
            return false;
        }
        const auto shape = leanShape(commands.subspan(cursor, Pm4::PacketWords(header)));
        if (!shape) {
            detail = aheadListWhere(commands, cursor) + ": today's walk finds a list the lean shape does not cover";
            return false;
        }
        if (index >= lists.refs.size()) {
            detail = aheadListWhere(commands, cursor) + ": the lists resolved ahead end before this list";
            return false;
        }
        const auto& ref = lists.refs[index++];
        if (ref.cursor != cursor || ref.first != first || ref.pairs != shape->count) {
            detail = aheadListWhere(commands, cursor) + ": the list resolved ahead sits at DWORD " + std::to_string(ref.cursor) + " word " + std::to_string(ref.first) + " with " + std::to_string(ref.pairs) + " pairs, today's at word " + std::to_string(first) + " with " + std::to_string(shape->count);
            return false;
        }
        fresh.assign(shape->Words(), 0);
        try {
            copyList(*shape, fresh.data());
        } catch (const std::exception& error) {
            detail = aheadListWhere(commands, cursor) + ", list " + hex(shape->address) + ": today's read failed: " + error.what();
            return false;
        }
        const bool valid = shape->count == 0 || Pm4::IndirectRegistersValid(fresh);
        if (((ref.flags & RegisterLists::Valid) != 0) != valid || (ref.flags & ~RegisterLists::Valid) != 0) {
            detail = aheadListWhere(commands, cursor) + ", list " + hex(shape->address) + ": flags " + hex(ref.flags) + " ahead, today's read is " + (valid ? "valid" : "invalid");
            return false;
        }
        if (ref.first + fresh.size() > lists.words.size()) {
            detail = aheadListWhere(commands, cursor) + ", list " + hex(shape->address) + ": the words resolved ahead end at " + std::to_string(lists.words.size()) + ", the list needs " + std::to_string(ref.first + fresh.size());
            return false;
        }
        const auto answer = std::span<const std::uint32_t>(lists.words).subspan(ref.first, fresh.size());
        const auto differs = std::mismatch(answer.begin(), answer.end(), fresh.begin());
        if (differs.first != answer.end()) {
            const auto at = static_cast<std::size_t>(differs.first - answer.begin());
            detail = aheadListWhere(commands, cursor) + ", list " + hex(shape->address) + " mode " + hex(shape->mode) + " count " + std::to_string(shape->count) + ": first difference at word " + std::to_string(at) + ": ahead " + hex(answer[at]) + ", today " + hex(fresh[at]);
            return false;
        }
        first += fresh.size();
    }
    if (index != lists.refs.size()) {
        detail = "the lists resolved ahead hold " + std::to_string(lists.refs.size()) + " lists, today's walk " + std::to_string(index);
        return false;
    }
    if (first != lists.words.size()) {
        detail = "the lists resolved ahead hold " + std::to_string(lists.words.size()) + " words, today's walk " + std::to_string(first);
        return false;
    }
    return true;
}

bool Driver::listsAheadWanted() const {
    if (countedReads() || listHooksVerifyEvery() != 0 || !frontLatch.switches.Set(Graphics::RuntimeSwitch::ListHooksOnce)) return false;
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::LeanListDirect) || !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::LeanListBatch);
}

bool Driver::takeAheadLists(Submission& submission, bool cacheable, AheadTail& job) {
    using TailsAhead::Count;
    const bool direct = cacheable && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::LeanListDirect);
    const bool batched = cacheable && !direct && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::LeanListBatch) && !countedReads();
    if (batched || countedReads() || !LeanFront::SwitchSet(frontLatch, Graphics::RuntimeSwitch::ListHooksOnce)) {
        TailsAhead::Add(Count::HooksWouldRun);
        return false;
    }
    if (listHooksVerifyEvery() != 0) {
        TailsAhead::Add(Count::Verify);
        return false;
    }
    const auto& spans = job.hookSpans;
    if (!spans.empty()) {
        bool all = Graphics::Recorder::FlushHookIdle(spans);
        for (std::size_t index = 0; all && index < spans.size(); ++index) all = Graphics::ReadGuardIdle(spans[index].first, static_cast<std::size_t>(spans[index].second - spans[index].first));
        if (!all) {
            TailsAhead::Add(Count::HooksWouldRun);
            return false;
        }
    }
    for (const auto& [begin, end] : job.collectSpans) {
        if (GuestMemory::CollectWritesUncached(begin, static_cast<std::size_t>(end - begin)) == 0) {
            TailsAhead::Add(Count::SourcesChanged);
            return false;
        }
    }
    if (!job.queries.empty() && !GuestMemory::UnchangedSinceAll(job.queries)) {
        GuestRanges ranges;
        ranges.reserve(job.queries.size());
        for (const auto& query : job.queries) ranges.emplace_back(query.address, query.bytes);
        TailsAhead::Add(GuestMemory::DriverStoredOver(ranges, job.queries.front().generation) ? Count::DriverStore : Count::SourcesChanged);
        return false;
    }
    for (const auto& [begin, end] : job.collectSpans) {
        if (GuestMemory::ForgetStamp(begin, static_cast<std::size_t>(end - begin)) > job.listForgets) {
            TailsAhead::Add(Count::ForgetMoved);
            return false;
        }
    }
    std::swap(submission.lists, job.resolved);
    submission.leanLists = true;
    if (direct) LeanFront::AddOnWorker(LeanFront::Counter::DirectListTails);
    LeanFront::AddOnWorker(LeanFront::Counter::LeanLists, job.listPackets);
    return true;
}

}
