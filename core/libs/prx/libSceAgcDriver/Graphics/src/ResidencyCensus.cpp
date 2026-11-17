#include "prx/libSceAgcDriver/Graphics/include/ResidencyCensus.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics::ResidencyCensus {

const bool Enabled = [] {
    const char* text = std::getenv("APS5_RESIDENCY_CENSUS");
    return text != nullptr && text[0] != '\0' && std::strcmp(text, "0") != 0;
}();

std::uint64_t LineSampleInterval() {
    static const std::uint64_t interval = [] {
        const char* text = std::getenv("APS5_RESIDENCY_CENSUS_LINES");
        return text != nullptr && text[0] != '\0' ? static_cast<std::uint64_t>(std::strtoull(text, nullptr, 10)) : std::uint64_t{4};
    }();
    return interval;
}

namespace {

constexpr std::size_t MaxRegions = 4096;
constexpr std::size_t MaxSamples = 256;
constexpr std::size_t MaxBuilds = 64;
constexpr std::size_t MaxPrograms = 8;
constexpr std::size_t MaxNames = 24;
constexpr std::size_t MaxTags = 24;
constexpr std::size_t ListedRegions = 32;
constexpr std::int64_t PeriodMs = 10000;
constexpr std::size_t SourceCount = static_cast<std::size_t>(GuestMemory::WriteBumpSource::Count);
constexpr std::size_t SiteCount = static_cast<std::size_t>(GuestMemory::ReadSite::Count);
constexpr std::size_t MissCount = static_cast<std::size_t>(Miss::Count);
constexpr std::array<const char*, SourceCount> SourceNames{"cpu", "driver", "pending", "unwatch", "forget", "release", "import"};

std::int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <typename... Args>
void append(std::string& line, const char* format, Args... args) {
    std::array<char, 640> text{};
    const auto size = std::snprintf(text.data(), text.size(), format, args...);
    if (size > 0) line.append(text.data(), std::min(static_cast<std::size_t>(size), text.size() - 1u));
}

double mib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / 1048576.0;
}

unsigned long long wide(std::uint64_t value) {
    return static_cast<unsigned long long>(value);
}

bool indexBreak() {
    static const bool broken = std::getenv("APS5_TABLE_INDEX_BREAK") != nullptr;
    return broken;
}

std::uint64_t writerProgram() {
    const auto tag = Recorder::CurrentWriterTag();
    return (tag >> 56u) == static_cast<std::uint64_t>(Stamper::Dispatch) ? tag & TagWhoMask : 0u;
}

struct TagCounts {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> entries;
    std::uint64_t other = 0;

    void Add(std::uint64_t tag, std::uint64_t count) {
        if (count == 0) return;
        for (auto& entry : entries) {
            if (entry.first != tag) continue;
            entry.second += count;
            return;
        }
        if (entries.size() >= MaxTags) {
            other += count;
            return;
        }
        entries.emplace_back(tag, count);
    }
};

template <typename Value>
struct Named {
    std::vector<std::pair<std::string, Value>> entries;
    std::uint64_t dropped = 0;

    Value* Find(const char* name) {
        const std::string_view key(name != nullptr ? name : "other");
        for (auto& entry : entries) {
            if (entry.first == key) return &entry.second;
        }
        if (entries.size() >= MaxNames) {
            ++dropped;
            return nullptr;
        }
        entries.emplace_back(std::string(key), Value{});
        return &entries.back().second;
    }
};

struct WriteBacks {
    std::uint64_t images = 0;
    std::uint64_t blockUnits = 0;
    std::uint64_t slab = 0;
    std::uint64_t imported = 0;
    std::uint64_t cpu = 0;
    std::uint64_t none = 0;
    std::uint64_t kept = 0;

    void Add(bool units, std::uint64_t slabBytes, std::uint64_t importBytes, std::uint64_t cpuBytes, std::uint64_t keptBlocks) {
        ++images;
        if (units) ++blockUnits;
        slab += slabBytes;
        imported += importBytes;
        cpu += cpuBytes;
        if (slabBytes == 0 && importBytes == 0 && cpuBytes == 0) ++none;
        kept += keptBlocks;
    }
};

struct Uploads {
    std::uint64_t count = 0;
    std::uint64_t bytes = 0;
};

struct Flushes {
    std::uint64_t flushes = 0;
    std::uint64_t images = 0;
    std::uint64_t slab = 0;
    std::uint64_t published = 0;
    std::uint64_t publishedUnits = 0;
};

struct Period {
    std::uint64_t uses = 0;
    std::uint64_t reused = 0;
    std::uint64_t fresh = 0;
    std::array<std::uint64_t, MissCount> step1{};
    std::array<std::uint64_t, MissCount> shared{};
    std::uint64_t intervals = 0;
    std::uint64_t clean = 0;
    std::array<std::uint64_t, SourceCount> events{};
    std::array<std::uint64_t, SourceCount> blocks{};
    std::array<std::uint64_t, SourceCount> own{};
    std::uint64_t cpuAtMarks = 0;
    std::uint64_t undecided = 0;
    std::array<std::uint64_t, SourceCount + 1u> first{};
    TagCounts firstTags;
    TagCounts driverTags;
    std::uint64_t followed = 0;
    std::uint64_t labels = 0;
    std::uint64_t flips = 0;
    std::uint64_t quiet = 0;
    std::array<std::uint64_t, SiteCount> hookReads{};
    std::uint64_t addressBuilds = 0;
    std::uint64_t usesWithImages = 0;
    Named<WriteBacks> writeBacks;
    Named<Uploads> uploads;
    std::array<std::uint64_t, SourceCount + 1u> uploadSources{};
    TagCounts uploadTags;
    std::uint64_t runs = 0;
    std::uint64_t samples = 0;
    std::uint64_t unwritten = 0;
    std::uint64_t lines = 0;
    std::uint64_t changedLines = 0;
    std::uint64_t changedWords = 0;
    std::vector<std::uint64_t> builds;
    std::vector<std::uint64_t> programs;
};

struct Region {
    std::uint32_t id = 0;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::shared_ptr<GuestMemory::WriteSubscription> subscription;
    std::vector<std::uint32_t> overlaps;
    bool seen = false;
    std::uint64_t build = 0;
    bool armed = false;
    bool armLayout = false;
    bool armCovered = false;
    bool armOwed = false;
    bool armGap = false;
    std::uint64_t useShadowSerial = 0;
    std::uint64_t sourceAddress = 0;
    std::uint64_t sourceBase = 0;
    GuestMemory::WriteCensusCounts atArm;
    GuestMemory::WriteCensusCounts atUse;
    bool undecided = false;
    UseFacts pendingFacts;
    std::uint64_t labelsAtUse = 0;
    std::uint64_t flipsAtUse = 0;
    bool imagesSinceUse = false;
    std::uint64_t runs = 0;
    Period period;
};

struct Sample {
    std::uint32_t region = 0;
    std::uint64_t recorder = 0;
    std::uint64_t serial = 0;
    std::uint64_t lines = 0;
    std::shared_ptr<Buffer> readback;
};

struct Drops {
    std::uint64_t count = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DropCause::Count)> causes{};
    std::uint64_t stampAfterPending = 0;
    std::uint64_t stampNotAfterPending = 0;
    std::uint64_t stampNoPendingSince = 0;
    std::uint64_t stampNotAfterUnit = 0;
    std::uint64_t inRegions = 0;
    std::array<std::uint64_t, SourceCount + 1u> sources{};
    TagCounts tags;
};

struct State {
    State() { samples.reserve(MaxSamples); }

    AgcDriver::Mutex mutex;
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::unique_ptr<Region>> regions;
    std::vector<Region*> byId;
    std::atomic<std::uint64_t> lowest{~std::uint64_t{0}};
    std::atomic<std::uint64_t> highest{0};
    std::atomic<std::uint64_t> builds{0};
    std::atomic<std::int64_t> copiesWindow{-1};
    std::uint64_t untracked = 0;
    std::int64_t periodStart = 0;
    std::uint64_t periodPresents = 0;
    std::vector<Sample> samples;
    Named<Flushes> flushes;
    Named<WriteBacks> writeBacks;
    Drops drops;
};

State& Census() {
    static auto* state = new State;
    return *state;
}

thread_local std::uint64_t threadSlabBytes = 0;

Region* regionById(State& state, std::uint32_t id) {
    return id != 0 && id <= state.byId.size() ? state.byId[id - 1u] : nullptr;
}

bool mayOverlap(const State& state, std::uint64_t begin, std::uint64_t end) {
    return begin < end && begin < state.highest.load(std::memory_order_relaxed) && state.lowest.load(std::memory_order_relaxed) < end;
}

template <typename Visit>
void eachOverlapping(State& state, std::uint64_t begin, std::uint64_t end, Visit&& visit) {
    for (auto* region : state.byId) {
        if (region->begin < end && begin < region->end) visit(*region);
    }
}

Region* regionFor(State& state, std::uint64_t begin, std::uint64_t end) {
    const auto key = std::make_pair(begin, end);
    if (const auto found = state.regions.find(key); found != state.regions.end()) return found->second.get();
    if (state.byId.size() >= MaxRegions) return nullptr;
    auto created = std::make_unique<Region>();
    created->id = static_cast<std::uint32_t>(state.byId.size() + 1u);
    created->begin = begin;
    created->end = end;
    for (auto* other : state.byId) {
        if (!(other->begin < end && begin < other->end)) continue;
        other->overlaps.push_back(created->id);
        created->overlaps.push_back(other->id);
    }
    auto* raw = created.get();
    state.byId.push_back(raw);
    state.regions.emplace(key, std::move(created));
    state.lowest.store(std::min(state.lowest.load(std::memory_order_relaxed), begin), std::memory_order_relaxed);
    state.highest.store(std::max(state.highest.load(std::memory_order_relaxed), end), std::memory_order_relaxed);
    return raw;
}

void noteIdentity(std::vector<std::uint64_t>& seen, std::uint64_t value, std::size_t limit) {
    if (std::find(seen.begin(), seen.end(), value) != seen.end() || seen.size() >= limit) return;
    seen.push_back(value);
}

void harvest(State& state, std::uint64_t recorder, std::uint64_t completedThrough, std::vector<std::shared_ptr<Buffer>>& released) {
    auto kept = state.samples.begin();
    for (auto it = state.samples.begin(); it != state.samples.end(); ++it) {
        if (it->recorder == recorder && it->serial <= completedThrough) {
            std::array<std::uint32_t, 2> counts{UnwrittenWord, UnwrittenWord};
            if (it->readback != nullptr && it->readback->Mapped()) std::memcpy(counts.data(), it->readback->Bytes().data(), sizeof(counts));
            if (auto* region = regionById(state, it->region)) {
                auto& period = region->period;
                if (counts[0] == UnwrittenWord && counts[1] == UnwrittenWord) {
                    ++period.unwritten;
                } else {
                    ++period.samples;
                    period.lines += it->lines;
                    period.changedLines += std::min<std::uint64_t>(counts[0], it->lines);
                    period.changedWords += counts[1];
                }
            }
            released.push_back(std::move(it->readback));
            continue;
        }
        if (kept != it) *kept = std::move(*it);
        ++kept;
    }
    state.samples.erase(kept, state.samples.end());
}

void tally(Period& period, const UseFacts& facts) {
    ++period.step1[static_cast<std::size_t>(FirstMiss(facts, false))];
    ++period.shared[static_cast<std::size_t>(FirstMiss(facts, true))];
}

void recordUse(Region& region, const Use& use, const GuestMemory::WriteCensusCounts& counts, std::uint64_t labels, std::uint64_t flips, std::uint64_t program) {
    auto& period = region.period;
    UseFacts facts;
    facts.seen = region.seen;
    facts.sameBuild = region.seen && region.build == use.build;
    facts.armed = region.armed;
    facts.copiedBack = (facts.sameBuild && use.reused) ? use.copiedBack : (!region.armed || region.armOwed);
    facts.gap = region.armGap;
    facts.layout = region.armLayout;
    facts.covered = region.armCovered;
    facts.sameImport = region.sourceAddress == use.sourceAddress && region.sourceBase == use.sourceBase;
    facts.watched = use.watched;
    facts.indexBreak = indexBreak();
    facts.bumped = region.armed && counts.generation != region.atArm.generation;
    facts.shadow = use.shadow;
    facts.pending = use.pending;
    facts.cpuWriter = use.cpuWriter;
    facts.queuedStore = use.queuedStore;
    ++period.uses;
    if (use.reused) ++period.reused;
    else ++period.fresh;
    if (region.undecided) {
        tally(period, region.pendingFacts);
        ++period.undecided;
    }
    region.pendingFacts = facts;
    region.undecided = true;
    region.atUse = counts;
    if (region.armed) {
        ++period.intervals;
        if (!facts.bumped) ++period.clean;
        for (std::size_t source = 0; source < SourceCount; ++source) {
            period.events[source] += counts.events[source] - region.atArm.events[source];
            period.blocks[source] += counts.blocks[source] - region.atArm.blocks[source];
        }
        const auto first = std::min(static_cast<std::size_t>(counts.firstSource), SourceCount);
        ++period.first[first];
        if (counts.firstSource == GuestMemory::WriteBumpSource::Driver || counts.firstSource == GuestMemory::WriteBumpSource::Pending) period.firstTags.Add(counts.firstTag, 1);
        for (const auto& [tag, total] : counts.tags) {
            std::uint64_t before = 0;
            for (const auto& [armTag, armTotal] : region.atArm.tags) {
                if (armTag == tag) before = armTotal;
            }
            period.driverTags.Add(tag, total - before);
        }
        period.driverTags.other += counts.otherTags - region.atArm.otherTags;
    }
    if (region.seen) {
        ++period.followed;
        const auto newLabels = labels - region.labelsAtUse;
        const auto newFlips = flips - region.flipsAtUse;
        period.labels += newLabels;
        period.flips += newFlips;
        if (newLabels == 0 && newFlips == 0) ++period.quiet;
    }
    if (region.imagesSinceUse) ++period.usesWithImages;
    region.imagesSinceUse = false;
    noteIdentity(period.builds, use.build, MaxBuilds);
    if (program != 0) noteIdentity(period.programs, program, MaxPrograms);
    region.seen = true;
    region.build = use.build;
    region.armed = false;
    region.armGap = false;
    region.useShadowSerial = use.shadowSerial;
    region.sourceAddress = use.sourceAddress;
    region.sourceBase = use.sourceBase;
    region.labelsAtUse = labels;
    region.flipsAtUse = flips;
}

std::string tagList(const TagCounts& tags) {
    if (tags.entries.empty() && tags.other == 0) return "none";
    auto sorted = tags.entries;
    std::sort(sorted.begin(), sorted.end(), [](const std::pair<std::uint64_t, std::uint64_t>& left, const std::pair<std::uint64_t, std::uint64_t>& right) { return left.second > right.second; });
    std::string text;
    for (const auto& [tag, total] : sorted) {
        if (!text.empty()) text += ',';
        append(text, "%s/0x%llx:%llu", StamperName(tag), wide(tag & TagWhoMask), wide(total));
    }
    if (tags.other != 0) {
        if (!text.empty()) text += ',';
        append(text, "other:%llu", wide(tags.other));
    }
    return text;
}

template <std::size_t Size>
std::string sourceList(const std::array<std::uint64_t, Size>& counts) {
    std::string text;
    for (std::size_t index = 0; index < Size; ++index) {
        if (counts[index] == 0) continue;
        if (!text.empty()) text += ',';
        append(text, "%s:%llu", index < SourceCount ? SourceNames[index] : "none", wide(counts[index]));
    }
    return text.empty() ? std::string("none") : text;
}

std::string missList(const std::array<std::uint64_t, MissCount>& counts) {
    std::string text;
    for (std::size_t index = 1; index < MissCount; ++index) {
        if (counts[index] == 0) continue;
        if (!text.empty()) text += ',';
        append(text, "%s:%llu", MissName(static_cast<Miss>(index)), wide(counts[index]));
    }
    return text.empty() ? std::string("none") : text;
}

std::string siteList(const std::array<std::uint64_t, SiteCount>& counts) {
    std::string text;
    for (std::size_t index = 0; index < SiteCount; ++index) {
        if (counts[index] == 0) continue;
        if (!text.empty()) text += ',';
        append(text, "%s:%llu", GuestMemory::ReadSiteName(static_cast<GuestMemory::ReadSite>(index)), wide(counts[index]));
    }
    return text.empty() ? std::string("none") : text;
}

std::string hexList(const std::vector<std::uint64_t>& values) {
    std::string text;
    for (const auto value : values) {
        if (!text.empty()) text += ',';
        append(text, "0x%llx", wide(value));
    }
    return text.empty() ? std::string("none") : text;
}

std::string idList(const std::vector<std::uint32_t>& ids) {
    std::string text;
    for (const auto id : ids) {
        if (!text.empty()) text += ',';
        append(text, "%u", static_cast<unsigned>(id));
    }
    return text.empty() ? std::string("none") : text;
}

std::string writeBackList(const Named<WriteBacks>& named) {
    std::string text;
    for (const auto& [reason, counts] : named.entries) {
        if (!text.empty()) text += ';';
        append(text, "\"%s\"{images=%llu,blockUnits=%llu,slabMiB=%.2f,importMiB=%.2f,cpuMiB=%.2f,none=%llu,keptBlocks=%llu}", reason.c_str(), wide(counts.images), wide(counts.blockUnits), mib(counts.slab), mib(counts.imported), mib(counts.cpu), wide(counts.none), wide(counts.kept));
    }
    if (named.dropped != 0) append(text, ";other{images=%llu}", wide(named.dropped));
    return text.empty() ? std::string("none") : text;
}

std::string uploadList(const Named<Uploads>& named) {
    std::string text;
    for (const auto& [reason, counts] : named.entries) {
        if (!text.empty()) text += ';';
        append(text, "\"%s\"{count=%llu,MiB=%.2f}", reason.c_str(), wide(counts.count), mib(counts.bytes));
    }
    if (named.dropped != 0) append(text, ";other{count=%llu}", wide(named.dropped));
    return text.empty() ? std::string("none") : text;
}

std::vector<std::string> report(State& state) {
    std::vector<std::string> lines;
    const auto now = nowMs();
    const auto presents = Recorder::Presents();
    if (state.periodStart == 0) {
        state.periodStart = now;
        state.periodPresents = presents;
        return lines;
    }
    if (now - state.periodStart < PeriodMs) return lines;
    const auto seconds = static_cast<double>(now - state.periodStart) / 1000.0;
    const auto frames = presents - state.periodPresents;
    const auto perPresent = [frames](std::uint64_t value) { return frames != 0 ? static_cast<double>(value) / static_cast<double>(frames) : 0.0; };
    std::vector<Region*> active;
    std::uint64_t uses = 0;
    for (auto* region : state.byId) {
        if (region->period.uses == 0) continue;
        active.push_back(region);
        uses += region->period.uses;
    }
    std::sort(active.begin(), active.end(), [](const Region* left, const Region* right) { return left->end - left->begin != right->end - right->begin ? left->end - left->begin > right->end - right->begin : left->id < right->id; });
    const auto listed = std::min(active.size(), ListedRegions);
    std::string header;
    append(header, "[residency-census] period seconds=%.1f presents=%llu regions=%zu active=%zu listed=%zu uses=%llu usesPerPresent=%.2f untrackedUses=%llu pendingLineSamples=%zu lineInterval=%llu", seconds, wide(frames), state.byId.size(), active.size(), listed, wide(uses), perPresent(uses), wide(state.untracked), state.samples.size(), wide(LineSampleInterval()));
    lines.push_back(std::move(header));
    for (std::size_t index = 0; index < listed; ++index) {
        const auto& region = *active[index];
        const auto& period = region.period;
        std::string line;
        append(line, "[residency-census] region id=%u base=0x%llx bytes=0x%llx uses=%llu usesPerPresent=%.2f reused=%llu new=%llu builds=%zu", static_cast<unsigned>(region.id), wide(region.begin), wide(region.end - region.begin), wide(period.uses), perPresent(period.uses), wide(period.reused), wide(period.fresh), period.builds.size());
        line += " programs=" + hexList(period.programs);
        line += " overlaps=" + idList(region.overlaps);
        append(line, " step1Hits=%llu step1HitsPerPresent=%.2f", wide(period.step1[0]), perPresent(period.step1[0]));
        line += " step1Misses=" + missList(period.step1);
        append(line, " sharedHits=%llu sharedHitsPerPresent=%.2f", wide(period.shared[0]), perPresent(period.shared[0]));
        line += " sharedMisses=" + missList(period.shared);
        lines.push_back(std::move(line));
        std::string bumps;
        append(bumps, "[residency-census] bumps id=%u intervals=%llu clean=%llu", static_cast<unsigned>(region.id), wide(period.intervals), wide(period.clean));
        bumps += " events=" + sourceList(period.events);
        bumps += " blocks=" + sourceList(period.blocks);
        bumps += " first=" + sourceList(period.first);
        bumps += " own=" + sourceList(period.own);
        append(bumps, " cpuAtMarks=%llu undecided=%llu", wide(period.cpuAtMarks), wide(period.undecided));
        bumps += " firstTags=" + tagList(period.firstTags);
        bumps += " driverTags=" + tagList(period.driverTags);
        append(bumps, " followed=%llu labels=%llu labelsPerInterval=%.2f flips=%llu quiet=%llu addressBuilds=%llu", wide(period.followed), wide(period.labels), period.followed != 0 ? static_cast<double>(period.labels) / static_cast<double>(period.followed) : 0.0, wide(period.flips), wide(period.quiet), wide(period.addressBuilds));
        bumps += " hookReads=" + siteList(period.hookReads);
        lines.push_back(std::move(bumps));
        std::string images;
        append(images, "[residency-census] images id=%u usesWithImages=%llu", static_cast<unsigned>(region.id), wide(period.usesWithImages));
        images += " writeBacks=" + writeBackList(period.writeBacks);
        images += " uploads=" + uploadList(period.uploads);
        images += " uploadStampers=" + sourceList(period.uploadSources);
        images += " uploadTags=" + tagList(period.uploadTags);
        lines.push_back(std::move(images));
        std::string sampled;
        const auto unchanged = period.lines - period.changedLines;
        append(sampled, "[residency-census] lines id=%u runs=%llu runsPerPresent=%.2f samples=%llu unwritten=%llu lines=%llu changedLines=%llu unchangedLines=%llu changedWords=%llu changedLineShare=%.4f", static_cast<unsigned>(region.id), wide(period.runs), perPresent(period.runs), wide(period.samples), wide(period.unwritten), wide(period.lines), wide(period.changedLines), wide(unchanged), wide(period.changedWords), period.lines != 0 ? static_cast<double>(period.changedLines) / static_cast<double>(period.lines) : 0.0);
        lines.push_back(std::move(sampled));
    }
    for (const auto& [reason, counts] : state.flushes.entries) {
        std::string line;
        append(line, "[residency-census] flush reason=\"%s\" flushes=%llu flushesPerPresent=%.2f images=%llu slabMiB=%.2f publishedAfterRetile=%llu publishedUnits=%llu", reason.c_str(), wide(counts.flushes), perPresent(counts.flushes), wide(counts.images), mib(counts.slab), wide(counts.published), wide(counts.publishedUnits));
        lines.push_back(std::move(line));
    }
    for (const auto& [reason, counts] : state.writeBacks.entries) {
        std::string line;
        append(line, "[residency-census] write-backs reason=\"%s\" images=%llu imagesPerPresent=%.2f blockUnits=%llu slabMiB=%.2f importMiB=%.2f cpuMiB=%.2f none=%llu keptBlocks=%llu", reason.c_str(), wide(counts.images), perPresent(counts.images), wide(counts.blockUnits), mib(counts.slab), mib(counts.imported), mib(counts.cpu), wide(counts.none), wide(counts.kept));
        lines.push_back(std::move(line));
    }
    {
        const auto& drops = state.drops;
        std::string line;
        append(line, "[residency-census] drops count=%llu clearKeys=%llu compared=%llu stamped=%llu stampAfterPending=%llu stampNotAfterPending=%llu stampNoPendingSince=%llu stampNotAfterUnit=%llu inRegions=%llu", wide(drops.count), wide(drops.causes[static_cast<std::size_t>(DropCause::ClearKeys)]), wide(drops.causes[static_cast<std::size_t>(DropCause::Compared)]), wide(drops.causes[static_cast<std::size_t>(DropCause::Stamped)]), wide(drops.stampAfterPending), wide(drops.stampNotAfterPending), wide(drops.stampNoPendingSince), wide(drops.stampNotAfterUnit), wide(drops.inRegions));
        line += " stampers=" + sourceList(drops.sources);
        line += " tags=" + tagList(drops.tags);
        lines.push_back(std::move(line));
    }
    for (auto* region : state.byId) region->period = Period{};
    state.flushes = Named<Flushes>{};
    state.writeBacks = Named<WriteBacks>{};
    state.drops = Drops{};
    state.untracked = 0;
    state.periodStart = now;
    state.periodPresents = presents;
    return lines;
}

void printLines(const std::vector<std::string>& lines) {
    for (const auto& line : lines) {
        try {
            AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
        } catch (...) {
            return;
        }
    }
}

std::string copyClass(const char* name, const CopyClass& counts) {
    std::string text;
    append(text, " %s=%llu/%.1fMiB", name, wide(counts.images), mib(counts.bytes));
    return text;
}

}

std::uint64_t StampScope::enter(Stamper kind, std::uint64_t who) noexcept {
    if (kind == Stamper::DirectMark) who = writerProgram();
    return GuestMemory::SetStampTag(Tag(kind, who));
}

void StampScope::leave(std::uint64_t tag) noexcept {
    GuestMemory::SetStampTag(tag);
}

std::uint64_t NewBuild() {
    return Census().builds.fetch_add(1, std::memory_order_relaxed) + 1u;
}

std::uint32_t NoteUse(const Use& use) {
    if (!On() || use.end <= use.begin) return 0;
    try {
        auto& state = Census();
        Region* region = nullptr;
        std::shared_ptr<GuestMemory::WriteSubscription> subscription;
        std::vector<std::shared_ptr<Buffer>> released;
        {
            std::lock_guard lock(state.mutex);
            harvest(state, use.recorder, use.completedThrough, released);
            region = regionFor(state, use.begin, use.end);
            if (region == nullptr) {
                ++state.untracked;
                return 0;
            }
            subscription = region->subscription;
        }
        released.clear();
        if (subscription == nullptr) {
            const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> range{{{use.begin, use.end}}};
            subscription = GuestMemory::SubscribeWriteCensus(range);
            std::lock_guard lock(state.mutex);
            region->subscription = subscription;
        }
        const auto counts = GuestMemory::ReadWriteCensus(*subscription, false);
        const auto labels = Recorder::QueuedStoreGeneration();
        const auto flips = Recorder::Presents();
        const auto program = writerProgram();
        std::vector<std::string> lines;
        {
            std::lock_guard lock(state.mutex);
            recordUse(*region, use, counts, labels, flips, program);
            lines = report(state);
        }
        printLines(lines);
        return region->id;
    } catch (...) {
        return 0;
    }
}

void NoteArm(const Arm& arm) {
    if (!On()) return;
    try {
        auto& state = Census();
        std::shared_ptr<GuestMemory::WriteSubscription> subscription;
        {
            std::lock_guard lock(state.mutex);
            const auto* found = regionById(state, arm.region);
            if (found == nullptr || found->build != arm.build) return;
            subscription = found->subscription;
        }
        if (subscription == nullptr) return;
        auto counts = GuestMemory::ReadWriteCensus(*subscription, true);
        std::lock_guard lock(state.mutex);
        auto* region = regionById(state, arm.region);
        if (region == nullptr || region->build != arm.build) return;
        if (region->undecided) {
            auto& period = region->period;
            for (std::size_t source = 0; source < SourceCount; ++source) period.own[source] += counts.events[source] - region->atUse.events[source];
            const auto cpu = static_cast<std::size_t>(GuestMemory::WriteBumpSource::Cpu);
            if (counts.events[cpu] != region->atUse.events[cpu]) {
                region->pendingFacts.cpuAtMarks = true;
                ++period.cpuAtMarks;
            }
            tally(period, region->pendingFacts);
            region->undecided = false;
        }
        region->armed = true;
        region->armLayout = arm.layout;
        region->armCovered = arm.covered;
        region->armOwed = arm.copyBackOwed;
        region->armGap = arm.shadowSerial != region->useShadowSerial;
        region->atArm = std::move(counts);
    } catch (...) {
    }
}

bool SampleLines(std::uint32_t region) {
    if (!On()) return false;
    const auto interval = LineSampleInterval();
    auto& state = Census();
    std::lock_guard lock(state.mutex);
    auto* entry = regionById(state, region);
    if (entry == nullptr) return false;
    ++entry->period.runs;
    if (interval == 0) return false;
    const bool due = entry->runs % interval == 0;
    ++entry->runs;
    return due && state.samples.size() < MaxSamples;
}

void NoteLineSample(std::uint32_t region, std::uint64_t recorder, std::uint64_t serial, std::uint64_t lines, std::shared_ptr<Buffer> readback) {
    if (!On() || readback == nullptr) return;
    auto& state = Census();
    std::shared_ptr<Buffer> dropped;
    {
        std::lock_guard lock(state.mutex);
        if (state.samples.size() < MaxSamples) state.samples.push_back(Sample{region, recorder, serial, lines, std::move(readback)});
        else dropped = std::move(readback);
    }
}

void NoteWriteBack(std::uint64_t begin, std::uint64_t end, const char* reason, bool blockUnits, std::uint64_t slabBytes, std::uint64_t importBytes, std::uint64_t cpuBytes, std::uint64_t keptBlocks) {
    if (!On()) return;
    threadSlabBytes += slabBytes;
    try {
        auto& state = Census();
        std::lock_guard lock(state.mutex);
        if (auto* totals = state.writeBacks.Find(reason)) totals->Add(blockUnits, slabBytes, importBytes, cpuBytes, keptBlocks);
        if (!mayOverlap(state, begin, end)) return;
        eachOverlapping(state, begin, end, [&](Region& region) {
            if (auto* counts = region.period.writeBacks.Find(reason)) counts->Add(blockUnits, slabBytes, importBytes, cpuBytes, keptBlocks);
            region.imagesSinceUse = true;
        });
    } catch (...) {
    }
}

std::uint64_t ThreadSlabBytes() {
    return threadSlabBytes;
}

void NoteFlush(const char* reason, std::size_t images, std::uint64_t slabBytes, std::uint64_t publishedUnits) {
    if (!On()) return;
    try {
        auto& state = Census();
        std::lock_guard lock(state.mutex);
        auto* counts = state.flushes.Find(reason);
        if (counts == nullptr) return;
        ++counts->flushes;
        counts->images += images;
        counts->slab += slabBytes;
        counts->publishedUnits += publishedUnits;
        if (slabBytes != 0 && publishedUnits != 0) ++counts->published;
    } catch (...) {
    }
}

void NoteUpload(std::uint64_t base, std::span<const std::pair<std::uint64_t, std::uint64_t>> runs, const char* reason) {
    if (!On()) return;
    try {
        struct Hit {
            std::uint32_t region = 0;
            std::shared_ptr<GuestMemory::WriteSubscription> subscription;
            std::uint64_t from = 0;
            std::uint64_t to = 0;
            GuestMemory::WriteStamper stamper;
        };
        auto& state = Census();
        std::vector<Hit> hits;
        {
            std::lock_guard lock(state.mutex);
            for (const auto& [first, last] : runs) {
                const auto begin = base + first;
                const auto end = base + last;
                if (!mayOverlap(state, begin, end)) continue;
                eachOverlapping(state, begin, end, [&](Region& region) { hits.push_back(Hit{region.id, region.subscription, std::max(begin, region.begin), std::min(end, region.end), {}}); });
            }
        }
        if (hits.empty()) return;
        for (auto& hit : hits) {
            if (hit.subscription != nullptr) hit.stamper = GuestMemory::NewestCensusStamper(*hit.subscription, hit.from, static_cast<std::size_t>(hit.to - hit.from));
        }
        std::lock_guard lock(state.mutex);
        for (const auto& hit : hits) {
            auto* region = regionById(state, hit.region);
            if (region == nullptr) continue;
            auto& period = region->period;
            if (auto* counts = period.uploads.Find(reason)) {
                ++counts->count;
                counts->bytes += hit.to - hit.from;
            }
            ++period.uploadSources[std::min(static_cast<std::size_t>(hit.stamper.source), SourceCount)];
            if (hit.stamper.source == GuestMemory::WriteBumpSource::Driver || hit.stamper.source == GuestMemory::WriteBumpSource::Pending) period.uploadTags.Add(hit.stamper.tag, 1);
        }
    } catch (...) {
    }
}

void NoteDrop(std::uint64_t begin, std::uint64_t bytes, std::uint64_t generation, std::uint64_t pendingSince, DropCause cause) {
    if (!On() || bytes == 0 || cause >= DropCause::Count) return;
    try {
        const bool stamped = cause == DropCause::Stamped;
        const auto newest = stamped ? GuestMemory::NewestStamp(begin, static_cast<std::size_t>(bytes)) : std::uint64_t{0};
        auto& state = Census();
        std::shared_ptr<GuestMemory::WriteSubscription> subscription;
        {
            std::lock_guard lock(state.mutex);
            if (mayOverlap(state, begin, begin + bytes)) {
                eachOverlapping(state, begin, begin + bytes, [&](Region& region) {
                    if (subscription == nullptr) subscription = region.subscription;
                });
            }
        }
        GuestMemory::WriteStamper stamper;
        if (subscription != nullptr) stamper = GuestMemory::NewestCensusStamper(*subscription, begin, static_cast<std::size_t>(bytes));
        std::lock_guard lock(state.mutex);
        auto& drops = state.drops;
        ++drops.count;
        ++drops.causes[static_cast<std::size_t>(cause)];
        if (stamped) {
            if (newest <= generation) ++drops.stampNotAfterUnit;
            if (pendingSince == 0) ++drops.stampNoPendingSince;
            else if (newest > pendingSince) ++drops.stampAfterPending;
            else ++drops.stampNotAfterPending;
        }
        if (subscription == nullptr) return;
        ++drops.inRegions;
        ++drops.sources[std::min(static_cast<std::size_t>(stamper.source), SourceCount)];
        if (stamper.source == GuestMemory::WriteBumpSource::Driver || stamper.source == GuestMemory::WriteBumpSource::Pending) drops.tags.Add(stamper.tag, 1);
    } catch (...) {
    }
}

void NoteHookRead(std::uint64_t address, std::size_t bytes, std::uint8_t site) {
    if (!On() || bytes == 0) return;
    auto& state = Census();
    const auto end = address + bytes;
    if (!mayOverlap(state, address, end)) return;
    std::lock_guard lock(state.mutex);
    const auto index = std::min<std::size_t>(site, SiteCount - 1u);
    eachOverlapping(state, address, end, [&](Region& region) { ++region.period.hookReads[index]; });
}

void NoteAddressBuild(std::uint64_t begin, std::uint64_t end) {
    if (!On()) return;
    auto& state = Census();
    if (!mayOverlap(state, begin, end)) return;
    std::lock_guard lock(state.mutex);
    eachOverlapping(state, begin, end, [](Region& region) { ++region.period.addressBuilds; });
}

bool CopiesDue() {
    if (!On()) return false;
    const auto window = nowMs() / PeriodMs;
    return Census().copiesWindow.exchange(window, std::memory_order_relaxed) != window;
}

void ReportCopies(const Copies& copies) {
    if (!On()) return;
    try {
        std::string line;
        append(line, "[residency-census] copies release=%s cached=%llu", copies.release ? "on" : "off", wide(copies.cachedImages));
        line += copyClass("holding", copies.holding);
        line += copyClass("holdsAll", copies.holdsAll);
        line += copyClass("holdsSome", copies.holdsSome);
        line += copyClass("holdsNone", copies.holdsNone);
        append(line, " released=%llu", wide(copies.released));
        line += copyClass("keptToVerify", copies.keptToVerify);
        line += copyClass("watched", copies.watched);
        line += copyClass("unwatched", copies.unwatched);
        line += copyClass("imported", copies.imported);
        line += copyClass("notImported", copies.notImported);
        line += copyClass("pending", copies.pending);
        line += copyClass("idle", copies.idle);
        line += copyClass("baselines", copies.baselines);
        line += copyClass("saved", copies.saved);
        constexpr std::array<const char*, static_cast<std::size_t>(CopyOrigin::Count)> origins{"unknown", "capture", "cpu-upload", "unwatch-save", "stored-save", "cpu-write-back", "restore"};
        line += " origins=";
        bool any = false;
        for (std::size_t index = 0; index < origins.size(); ++index) {
            const auto& counts = copies.origins[index];
            if (counts.images == 0) continue;
            if (any) line += ',';
            any = true;
            append(line, "%s:%llu/%.1fMiB", origins[index], wide(counts.images), mib(counts.bytes));
        }
        if (!any) line += "none";
        printLines({line});
    } catch (...) {
    }
}

}
