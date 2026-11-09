#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/StagePlans.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/LeanDraw.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::uint64_t FnvPrime = 0x100000001b3ull;

std::uint64_t AlignedWord(std::uint32_t word) {
    return word & ~std::uint32_t{3};
}

std::uint32_t LowBits(std::uint32_t word) {
    return (word & 3u) | (word != 0u ? 4u : 0u);
}

std::vector<std::uint32_t> VertexLayoutWords(const std::optional<ShaderRecompiler::ShaderVertexStageInfo>& info) {
    std::vector<std::uint32_t> words;
    if (!info) return words;
    const auto count = std::min<std::uint32_t>(info->resourcesNum, ShaderRecompiler::ShaderVertexStageInfo::MaxResources);
    words = {0x80000000u | count, info->fetchAttribReg, info->fetchBufferReg, info->fetchEmbedded ? 1u : 0u, info->paClVsOutCntl, info->depthRemapMode, info->pointList ? 1u : 0u, std::bit_cast<std::uint32_t>(info->pointSize), std::bit_cast<std::uint32_t>(info->pointSizeMin), std::bit_cast<std::uint32_t>(info->pointSizeMax)};
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto& fields = info->resources[index].fields;
        const auto& destination = info->resourcesDst[index];
        words.insert(words.end(), {fields[1] & 0xffff0000u, fields[3], static_cast<std::uint32_t>(destination.registerStart), static_cast<std::uint32_t>(destination.registersNum), static_cast<std::uint32_t>(destination.attrId), destination.fetchIndex});
    }
    return words;
}

bool SameStagePlan(const StagePlan& kept, const StagePlan& built) {
    const auto& a = kept.stage;
    const auto& b = built.stage;
    if (a.compiled == nullptr || b.compiled == nullptr || a.compiled->variantId != b.compiled->variantId || a.pushConstantOffset != b.pushConstantOffset || a.stage != b.stage) return false;
    if (kept.vertexLayout != built.vertexLayout) return false;
    const bool viewsA = a.registeredViews != nullptr && !a.registeredViews->empty();
    const bool viewsB = b.registeredViews != nullptr && !b.registeredViews->empty();
    if (viewsA != viewsB) return false;
    return !viewsA || *a.registeredViews == *b.registeredViews;
}

std::size_t LayoutBytes(const StageLayout& layout) {
    return sizeof(StageLayout) + sizeof(StageRun) * layout.runs.capacity() + sizeof(std::uint32_t) * (layout.keyPositions.capacity() + layout.bitPositions.capacity());
}

std::size_t StageBytes(const DrawRecordStage& stage) {
    std::size_t bytes = sizeof(DrawRecordStage) + sizeof(std::uint32_t) * stage.userData.capacity() + sizeof(DrawRecordTarget) * stage.targets.capacity() + sizeof(std::array<std::uint32_t, 3>) * (stage.valueSlots.capacity() + stage.baseValueSlots.capacity()) + sizeof(ShaderRecompiler::MemoryRegion) * stage.regions.capacity();
    if (stage.variant != nullptr) bytes += sizeof(DispatchVariant) + sizeof(std::uint32_t) * (stage.variant->words.capacity() + stage.variant->dataPositions.capacity()) + sizeof(std::pair<std::uint64_t, std::uint64_t>) * stage.variant->runs.capacity();
    if (stage.compiled != nullptr) {
        bytes += sizeof(ShaderRecompiler::RecompileResult) + stage.compiled->pushConstants.size();
        for (const auto& binding : stage.compiled->bindings) bytes += sizeof(binding) + sizeof(std::uint32_t) * binding.guestDescriptor.size();
    }
    return bytes;
}

}

std::size_t StagePlanSet::PlanCount() const {
    std::size_t count = 0;
    for (const auto& stage : stages) count += stage.plans.size();
    return count;
}

std::shared_ptr<const StagePlan> MakeStagePlan(const DrawRecordStage& stage) {
    using DescriptorRole = ShaderRecompiler::DescriptorRole;
    if (stage.variant == nullptr || stage.compiled == nullptr) return nullptr;
    const auto& variant = *stage.variant;
    const auto& compiled = *stage.compiled;
    const auto users = stage.userData.size();
    if (users > 64u) return nullptr;
    auto layout = std::make_shared<StageLayout>();
    layout->users = static_cast<std::uint32_t>(users);
    std::size_t total = 0;
    for (std::size_t run = 0; run < variant.runs.size(); ++run) {
        const auto [begin, end] = variant.runs[run];
        if (end < begin || (end - begin) % sizeof(std::uint32_t) != 0) return nullptr;
        StageRun entry;
        entry.words = static_cast<std::uint32_t>((end - begin) / sizeof(std::uint32_t));
        const auto base = run < stage.runBases.size() ? stage.runBases[run] : std::int8_t{-1};
        if (base >= 0) {
            if (static_cast<std::size_t>(base) >= users) return nullptr;
            entry.base = base;
            entry.offset = begin - AlignedWord(stage.userData[static_cast<std::size_t>(base)]);
        } else {
            entry.offset = begin;
        }
        layout->runs.push_back(entry);
        total += entry.words;
    }
    if (total != variant.words.size()) return nullptr;
    layout->words = static_cast<std::uint32_t>(total);
    std::vector<std::uint8_t> kinds(total, 0);
    for (const auto& target : stage.targets) {
        if (target.position >= total) return nullptr;
        kinds[target.position] |= 1u;
        if (target.binding < compiled.bindings.size() && compiled.bindings[target.binding].role == DescriptorRole::GuestBuffers) kinds[target.position] |= 2u;
    }
    for (std::uint32_t position = 0; position < total; ++position) {
        if ((kinds[position] & 1u) == 0) layout->keyPositions.push_back(position);
        else if ((kinds[position] & 2u) != 0) layout->bitPositions.push_back(position);
    }
    const auto moving = stage.relocatableWords | stage.baseValueWords | stage.valueWords;
    for (std::uint32_t index = 0; index < users; ++index) {
        const auto bit = std::uint64_t{1} << index;
        if (index < stage.hiddenUserWords) layout->exactUsers |= bit;
        else if ((moving & bit) != 0) layout->bitUsers |= bit;
        else if ((stage.walkedWords & bit) != 0) layout->exactUsers |= bit;
        else if (((stage.vertexWords | stage.forwardedWords) & bit) == 0 && (stage.userDataMask & bit) != 0) layout->exactUsers |= bit;
    }
    auto plan = std::make_shared<StagePlan>();
    plan->layout = layout;
    plan->stage = stage;
    plan->vertexLayout = VertexLayoutWords(stage.vertexInfo);
    plan->addressed = std::ranges::any_of(compiled.bindings, [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == DescriptorRole::BdaPagetable; });
    StageKeyOf(*layout, stage.userData, variant.words, plan->key);
    plan->bytes = sizeof(StagePlan) + LayoutBytes(*layout) + sizeof(std::uint32_t) * (plan->key.words.capacity() + plan->vertexLayout.capacity()) + StageBytes(stage);
    return plan;
}

void StageKeyOf(const StageLayout& layout, std::span<const std::uint32_t> user, std::span<const std::uint32_t> words, StagePlanKey& key) {
    key.words.clear();
    key.words.reserve(static_cast<std::size_t>(std::popcount(layout.exactUsers) + std::popcount(layout.bitUsers)) + layout.keyPositions.size() + layout.bitPositions.size());
    const auto userWord = [&](std::size_t index) { return index < user.size() ? user[index] : 0u; };
    for (auto bits = layout.exactUsers; bits != 0; bits &= bits - 1) key.words.push_back(userWord(static_cast<std::size_t>(std::countr_zero(bits))));
    for (auto bits = layout.bitUsers; bits != 0; bits &= bits - 1) key.words.push_back(LowBits(userWord(static_cast<std::size_t>(std::countr_zero(bits)))));
    for (const auto position : layout.keyPositions) key.words.push_back(position < words.size() ? words[position] : 0u);
    for (const auto position : layout.bitPositions) key.words.push_back(LowBits(position < words.size() ? words[position] : 0u));
    std::uint64_t hash = 0xcbf29ce484222325ull ^ key.words.size();
    for (const auto word : key.words) hash = (hash ^ word) * FnvPrime;
    key.hash = hash == 0 ? 1 : hash;
}

bool ReadStageRuns(const StageLayout& layout, std::span<const std::uint32_t> user, std::vector<std::uint32_t>& words) {
    words.resize(layout.words);
    std::size_t at = 0;
    for (const auto& run : layout.runs) {
        auto address = run.offset;
        if (run.base >= 0) {
            if (static_cast<std::size_t>(run.base) >= user.size()) return false;
            address += AlignedWord(user[static_cast<std::size_t>(run.base)]);
        }
        if (run.words != 0 && GuestMemory::CopyMapped(address, std::as_writable_bytes(std::span(words.data() + at, run.words))) != GuestMemory::Compare::Equal) return false;
        at += run.words;
    }
    return true;
}

const StagePlan* FindStagePlan(const StagePlanStage& stage, const StageLayout* layout, const StagePlanKey& key) {
    for (const auto& plan : stage.plans) {
        if (plan->layout.get() == layout && plan->key.hash == key.hash && plan->key.words == key.words) return plan.get();
    }
    return nullptr;
}

StagePlans::Counter StageKeyMiss(const StagePlanStage& stage, const StageLayout* layout, const StagePlanKey& key) {
    using StagePlans::Counter;
    const auto newest = std::find_if(stage.plans.begin(), stage.plans.end(), [&](const auto& plan) { return plan->layout.get() == layout; });
    if (newest == stage.plans.end() || layout == nullptr) return Counter::MissNewRunWords;
    const auto& kept = (*newest)->key.words;
    if (kept.size() != key.words.size()) return Counter::MissNewRunWords;
    const auto exact = static_cast<std::size_t>(std::popcount(layout->exactUsers));
    const auto bits = static_cast<std::size_t>(std::popcount(layout->bitUsers));
    const auto runs = layout->keyPositions.size();
    const auto differs = [&](std::size_t from, std::size_t count) { return !std::equal(kept.begin() + static_cast<std::ptrdiff_t>(from), kept.begin() + static_cast<std::ptrdiff_t>(from + count), key.words.begin() + static_cast<std::ptrdiff_t>(from)); };
    if (differs(0, exact)) return Counter::MissNewUserWords;
    if (differs(exact + bits, runs)) return Counter::MissNewRunWords;
    return Counter::MissNewBits;
}

const DrawPlan* FindDrawPlan(const StagePlanSet& set, std::span<const StagePlan* const> stages) {
    for (const auto& draw : set.draws) {
        if (draw.stages.size() != stages.size()) continue;
        bool same = true;
        for (std::size_t i = 0; i < stages.size() && same; ++i) same = draw.stages[i].get() == stages[i];
        if (same) return &draw;
    }
    return nullptr;
}

std::shared_ptr<const DrawRecord> AssembleDrawRecord(const DrawRecord& built, std::span<const std::shared_ptr<const StagePlan>> stages) {
    auto record = std::make_shared<DrawRecord>();
    record->key = built.key;
    record->probe = built.probe;
    record->shape = built.shape;
    record->deviceSerial = built.deviceSerial;
    record->generation = built.generation;
    record->builtPresent = built.builtPresent;
    record->content = built.content;
    record->inputs = built.inputs;
    record->contentKey = built.contentKey;
    record->vertexLayout = built.vertexLayout;
    record->vertexInputs = built.vertexInputs;
    record->fragmentOutputs = built.fragmentOutputs;
    record->writes = built.writes;
    auto plan = std::make_shared<DrawRecordPlan>();
    plan->stages.reserve(stages.size());
    for (const auto& stage : stages) {
        plan->stages.push_back(stage->stage);
        plan->addressed = plan->addressed || stage->addressed;
        plan->bytes += stage->bytes;
    }
    plan->signature = 1;
    record->plan = std::move(plan);
    record->signature = 1;
    return record;
}

std::shared_ptr<StagePlanSet> MergeStagePlans(const StagePlanSet* kept, const StagePlanAnswer& answer, StagePlanMerge& merge) {
    using namespace StagePlanLimits;
    const auto count = answer.stages.size();
    if (count == 0 || answer.record == nullptr || answer.shape == nullptr) return nullptr;
    if (std::any_of(answer.stages.begin(), answer.stages.end(), [](const auto& stage) { return stage == nullptr || stage->layout == nullptr; })) return nullptr;
    auto set = std::make_shared<StagePlanSet>();
    if (kept != nullptr && kept->shape == answer.shape && kept->generation == answer.generation && kept->deviceSerial == answer.deviceSerial && kept->stages.size() == count) {
        set->stages = kept->stages;
        set->draws = kept->draws;
    } else {
        set->stages.resize(count);
    }
    set->shape = answer.shape;
    set->generation = answer.generation;
    set->deviceSerial = answer.deviceSerial;
    std::vector<std::shared_ptr<const StagePlan>> chosen(count);
    std::vector<const StagePlan*> dropped;
    for (std::size_t i = 0; i < count; ++i) {
        auto& stage = set->stages[i];
        const auto& built = *answer.stages[i];
        std::shared_ptr<const StageLayout> layout;
        for (const auto& known : stage.layouts) {
            if (*known == *built.layout) {
                layout = known;
                break;
            }
        }
        if (layout == nullptr) {
            layout = built.layout;
            stage.layouts.insert(stage.layouts.begin(), layout);
            ++merge.layouts;
        }
        const auto same = std::find_if(stage.plans.begin(), stage.plans.end(), [&](const auto& plan) { return plan->layout == layout && plan->key == built.key; });
        if (same != stage.plans.end() && SameStagePlan(**same, built)) {
            chosen[i] = *same;
            ++merge.shared;
        } else {
            auto plan = std::make_shared<StagePlan>(built);
            plan->layout = layout;
            if (same != stage.plans.end()) {
                dropped.push_back(same->get());
                stage.plans.erase(same);
                ++merge.replaced;
            }
            stage.plans.insert(stage.plans.begin(), plan);
            chosen[i] = std::move(plan);
            ++merge.added;
            while (stage.plans.size() > PlansPerStage) {
                dropped.push_back(stage.plans.back().get());
                stage.plans.pop_back();
                ++merge.evicted;
            }
        }
        while (stage.layouts.size() > Layouts) {
            const auto* old = stage.layouts.back().get();
            std::erase_if(stage.plans, [&](const auto& plan) {
                if (plan->layout.get() != old) return false;
                dropped.push_back(plan.get());
                ++merge.evicted;
                return true;
            });
            stage.layouts.pop_back();
        }
        std::erase_if(stage.layouts, [&](const auto& known) { return std::none_of(stage.plans.begin(), stage.plans.end(), [&](const auto& plan) { return plan->layout == known; }); });
    }
    if (!dropped.empty()) {
        const auto before = set->draws.size();
        std::erase_if(set->draws, [&](const DrawPlan& draw) { return std::any_of(draw.stages.begin(), draw.stages.end(), [&](const auto& stage) { return std::find(dropped.begin(), dropped.end(), stage.get()) != dropped.end(); }); });
        merge.drawsEvicted += before - set->draws.size();
    }
    std::vector<const StagePlan*> tuple;
    tuple.reserve(count);
    for (const auto& plan : chosen) tuple.push_back(plan.get());
    if (FindDrawPlan(*set, tuple) != nullptr) {
        ++merge.drawsKnown;
    } else {
        DrawPlan draw;
        draw.stages = chosen;
        draw.record = AssembleDrawRecord(*answer.record, chosen);
        draw.bytes = sizeof(DrawPlan) + draw.record->Bytes() - draw.record->plan->bytes + sizeof(DrawRecordStage) * count;
        set->draws.insert(set->draws.begin(), std::move(draw));
        ++merge.drawsAdded;
        while (set->draws.size() > DrawPlans) {
            set->draws.pop_back();
            ++merge.drawsEvicted;
        }
    }
    set->bytes = sizeof(StagePlanSet);
    for (const auto& stage : set->stages) {
        for (const auto& layout : stage.layouts) set->bytes += LayoutBytes(*layout);
        for (const auto& plan : stage.plans) set->bytes += plan->bytes;
    }
    for (const auto& draw : set->draws) set->bytes += draw.bytes;
    return set;
}

namespace StagePlans {

namespace {

using DrawRecords::Allowed;
using DrawRecords::Field;
using DrawRecords::Item;

constexpr std::size_t FieldCount = static_cast<std::size_t>(Field::Count);
constexpr std::size_t AllowedKinds = static_cast<std::size_t>(Allowed::Count);
constexpr std::size_t ItemCount = static_cast<std::size_t>(Item::Count);
constexpr std::size_t CounterCount = static_cast<std::size_t>(Counter::Count);
constexpr std::uint64_t PrintedPerWindow = 8;
constexpr std::int64_t WindowMs = 10000;

constexpr std::array<const char*, CounterCount> CounterNames{
    "draws", "hits", "hits on today's path", "input refetches", "relocated hits", "relocated base and value words", "moved buffer bases patched",
    "no plans", "refused", "generation moved", "user words unreadable", "new layout", "runs unreadable", "new user words", "new low bits", "new run words", "new combination",
    "image views moved", "content differs", "inaccessible", "queued label", "flushing image", "forget serial moved", "publish moved", "pending storage moved", "device images", "moved buffer fields", "vertex layout differs", "memory changed", "fetch error",
    "today's draw", "indirect", "lean front verify", "inline",
    "builds", "not queued", "generation moved", "shape", "indirect", "stage count", "geometry back end", "capture", "direct stage", "program", "result", "patched programs", "layout",
    "builds dropped", "dropped without an answer slot", "dropped as stale", "dropped for a replaced shape", "dropped for a refused shape", "dropped over the byte cap",
    "stage plans added", "stage plans shared", "stage plans replaced", "stage plans evicted", "layouts added", "draw plans added", "draw plans known", "draw plans evicted", "sets swept", "sets cleared", "mode changes", "demotions",
};

constexpr std::array<const char*, AllowedKinds> AllowedNames{"fragment outputs known on one side only", "today's draw not queued", "the record's commit fell back", "today's commit built no recipe", "today's commit served another template"};

static_assert(CounterNames.back() != nullptr && AllowedNames.back() != nullptr, "every stage plan counter and by-design case needs a name");

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
std::atomic<std::uint64_t> storeShapes{0};
std::atomic<std::uint64_t> storePlans{0};
std::atomic<std::uint64_t> storeDraws{0};
std::atomic<std::uint64_t> storeBytes{0};
std::mutex reportMutex;
std::int64_t reportedAt = 0;

bool Printed() {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    return stats;
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

}

std::uint64_t VerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_STAGE_PLANS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

void NoteChecked(Item item) {
    checked[static_cast<std::size_t>(item)].value.fetch_add(1, std::memory_order_relaxed);
}

void NoteDifference(Field field, std::string_view detail) {
    differences[static_cast<std::size_t>(field)].value.fetch_add(1, std::memory_order_relaxed);
    if (printed.fetch_add(1, std::memory_order_relaxed) < PrintedPerWindow) std::fprintf(stderr, "[stage-plans-verify] %s %s differs: %.*s\n", DrawRecords::ItemName(Item::Fetch), DrawRecords::FieldName(field), static_cast<int>(detail.size()), detail.data());
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

void NoteStore(std::size_t shapes, std::size_t stagePlans, std::size_t drawPlans, std::size_t bytes) {
    storeShapes.store(shapes, std::memory_order_relaxed);
    storePlans.store(stagePlans, std::memory_order_relaxed);
    storeDraws.store(drawPlans, std::memory_order_relaxed);
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
    const auto at = [&](Counter counter) { return counts[static_cast<std::size_t>(counter)]; };
    const auto draws = at(Counter::Draws);
    if (Printed() && (draws != 0 || at(Counter::ModeChanges) != 0 || at(Counter::SetsCleared) != 0)) {
        const auto hits = at(Counter::Hits);
        AgcDriver::ProfilePrint_nid_no_patch("[stage-plans] %.1f s: draws %llu, hits %llu (%.1f%%), hits on today's path %llu, input refetches %llu, relocated hits %llu, relocated base and value words %llu, moved buffer bases patched %llu; misses: %s; plans off: %s; builds: %s; store: %s; %llu shapes, %llu stage plans, %llu draw plans, %.1f KiB\n", seconds,
            static_cast<unsigned long long>(draws), static_cast<unsigned long long>(hits), draws != 0 ? 100.0 * static_cast<double>(hits) / static_cast<double>(draws) : 0.0,
            static_cast<unsigned long long>(at(Counter::HitsOnTodaysPath)), static_cast<unsigned long long>(at(Counter::InputRefetches)), static_cast<unsigned long long>(at(Counter::HitsRelocated)),
            static_cast<unsigned long long>(at(Counter::HitsBaseValue)), static_cast<unsigned long long>(at(Counter::HitsBuffersPatched)),
            counted(counts, Counter::MissNoPlans, Counter::MissFetchError).c_str(), counted(counts, Counter::OffTodaysDraw, Counter::OffInline).c_str(), counted(counts, Counter::Builds, Counter::DroppedOverCap).c_str(), counted(counts, Counter::StagePlansAdded, Counter::Demotions).c_str(),
            static_cast<unsigned long long>(storeShapes.load(std::memory_order_relaxed)), static_cast<unsigned long long>(storePlans.load(std::memory_order_relaxed)), static_cast<unsigned long long>(storeDraws.load(std::memory_order_relaxed)), static_cast<double>(storeBytes.load(std::memory_order_relaxed)) / 1024.0);
    }
    if (VerifyEvery() == 0 || (draws == 0 && checksTotal == 0 && differingTotal == 0 && byDesignTotal == 0)) return;
    const auto fields = listed(FieldCount, [&](std::size_t index) { return differing[index]; }, [&](std::size_t index) { return std::string(DrawRecords::FieldName(static_cast<Field>(index))); });
    const auto cases = listed(AllowedKinds, [&](std::size_t index) { return byDesign[index]; }, [&](std::size_t index) { return std::string(AllowedNames[index]); });
    AgcDriver::ProfilePrint_nid_no_patch("[stage-plans-verify] %.1f s: every %llu; checked: hits %llu; differences %llu%s; by design %llu%s\n", seconds, static_cast<unsigned long long>(VerifyEvery()), static_cast<unsigned long long>(checks[static_cast<std::size_t>(Item::Fetch)]), static_cast<unsigned long long>(differingTotal), fields.c_str(), static_cast<unsigned long long>(byDesignTotal), cases.c_str());
}

const char* CounterName(Counter counter) {
    return CounterNames[static_cast<std::size_t>(counter)];
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

namespace {

StagePlans::Counter FetchMiss(DrawRecords::Counter outcome, EntryOutcome gate) {
    using StagePlans::Counter;
    switch (outcome) {
    case DrawRecords::Counter::MissNoFetchPlan: return Counter::MissNoPlans;
    case DrawRecords::Counter::MissUserWordUnread: return Counter::MissUserWordsUnread;
    case DrawRecords::Counter::MissImageViews: return Counter::MissImageViews;
    case DrawRecords::Counter::MissContentDiffers: return Counter::MissContentDiffers;
    case DrawRecords::Counter::MissBufferFields: return Counter::MissBufferFields;
    case DrawRecords::Counter::MissVertexLayout: return Counter::MissVertexLayout;
    case DrawRecords::Counter::MissMemoryChanged: return Counter::MissMemoryChanged;
    case DrawRecords::Counter::MissFetchGate:
        switch (gate) {
        case EntryOutcome::Inaccessible: return Counter::MissInaccessible;
        case EntryOutcome::QueuedLabel: return Counter::MissQueuedLabel;
        case EntryOutcome::FlushingImage: return Counter::MissFlushingImage;
        case EntryOutcome::PublishMoved: return Counter::MissPublishMoved;
        case EntryOutcome::PendingMoved: return Counter::MissPendingMoved;
        case EntryOutcome::ForgetMoved: return Counter::MissForgetMoved;
        case EntryOutcome::DeviceImageFallback: return Counter::MissDeviceImages;
        default: return Counter::MissFetchError;
        }
    default: return Counter::MissFetchError;
    }
}

StagePlans::Counter Refusal(DrawRecordPlanRefusal refusal) {
    using StagePlans::Counter;
    switch (refusal) {
    case DrawRecordPlanRefusal::Shape: return Counter::RefusedShape;
    case DrawRecordPlanRefusal::Indirect: return Counter::RefusedIndirect;
    case DrawRecordPlanRefusal::Stages: return Counter::RefusedStages;
    case DrawRecordPlanRefusal::GeometryBack: return Counter::RefusedGeometryBack;
    case DrawRecordPlanRefusal::Capture: return Counter::RefusedCapture;
    case DrawRecordPlanRefusal::Direct: return Counter::RefusedDirect;
    case DrawRecordPlanRefusal::Program: return Counter::RefusedProgram;
    case DrawRecordPlanRefusal::Result: return Counter::RefusedResult;
    case DrawRecordPlanRefusal::Patched: return Counter::RefusedPatched;
    default: return Counter::RefusedCapture;
    }
}

bool BuildsOn(StagePlans::Counter miss) {
    using StagePlans::Counter;
    switch (miss) {
    case Counter::MissNoPlans:
    case Counter::MissNewLayout:
    case Counter::MissRunsUnreadable:
    case Counter::MissNewUserWords:
    case Counter::MissNewBits:
    case Counter::MissNewRunWords:
    case Counter::MissNewCombination:
    case Counter::MissGenerationMoved:
    case Counter::MissImageViews:
    case Counter::MissBufferFields:
    case Counter::MissVertexLayout: return true;
    default: return false;
    }
}

struct PlanScratchStorage {};

struct PlanScratch {
    std::vector<std::vector<std::uint32_t>> users;
    std::vector<std::uint32_t> words;
    StagePlanKey key;
    std::vector<const StagePlan*> found;
};

}

void Driver::probeStagePlans(LeanDrawJob& job, ShapeRecord& record) {
    using StagePlans::Counter;
    job.recordFlags = DrawRecordFlags::StagePlans;
    job.recordGeneration = frontLatch.generation;
    job.records = nullptr;
    job.plans = nullptr;
    StagePlans::AddOnWorker(Counter::Draws);
    if (record.plansRefused) {
        StagePlans::AddOnWorker(Counter::MissRefused);
        return;
    }
    job.recordFlags |= DrawRecordFlags::BuildWanted;
    if (record.plans == nullptr) return;
    job.plans = record.plans.get();
    record.plansSeq = DrawFinisher::Get().Issued();
    record.plansPresent = Graphics::Recorder::Presents();
    if (StagePlans::Due(frontLatch)) job.recordFlags |= DrawRecordFlags::VerifyDue;
}

std::shared_ptr<DrawRecordHit> Driver::lookupStagePlans(const LeanDrawJob& job, StagePlans::Counter& miss) {
    using StagePlans::Counter;
    DrawPipeline::FollowEpoch(job.epoch);
    DrawPipeline::Active() = true;
    GuestMemory::SetCurrentPacket(job.opcode, job.queue);
    const DrawFinisher::AheadScope ahead;
    const auto& set = *job.plans;
    if (set.generation != job.recordGeneration || set.deviceSerial != job.deviceSerial || set.shape.get() != job.shape) {
        miss = Counter::MissGenerationMoved;
        return nullptr;
    }
    const auto& programs = job.shape->decode->programs;
    const auto count = programs.size();
    if (set.stages.size() != count || count == 0) {
        miss = Counter::MissNoPlans;
        return nullptr;
    }
    auto& scratch = HostThreadLocal<PlanScratch, PlanScratchStorage>();
    if (scratch.users.size() < count) scratch.users.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (ReadRecordUserWords(programs[i], job.words, scratch.users[i])) continue;
        miss = Counter::MissUserWordsUnread;
        return nullptr;
    }
    scratch.found.assign(count, nullptr);
    const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& stage = set.stages[i];
        const auto& user = scratch.users[i];
        auto reason = Counter::MissNewLayout;
        for (const auto& layout : stage.layouts) {
            if (layout->users != user.size()) continue;
            if (!ReadStageRuns(*layout, user, scratch.words)) {
                if (reason == Counter::MissNewLayout) reason = Counter::MissRunsUnreadable;
                continue;
            }
            StageKeyOf(*layout, user, scratch.words, scratch.key);
            if (const auto* plan = FindStagePlan(stage, layout.get(), scratch.key)) {
                scratch.found[i] = plan;
                break;
            }
            if (reason == Counter::MissNewLayout || reason == Counter::MissRunsUnreadable) reason = StageKeyMiss(stage, layout.get(), scratch.key);
        }
        if (scratch.found[i] != nullptr) continue;
        miss = reason;
        return nullptr;
    }
    const auto* draw = FindDrawPlan(set, scratch.found);
    if (draw == nullptr) {
        miss = Counter::MissNewCombination;
        return nullptr;
    }
    auto hit = std::make_shared<DrawRecordHit>();
    auto gate = EntryOutcome::Equal;
    DrawRecords::Counter outcome;
    try {
        outcome = fetchRecordPlan(job, *draw->record->plan, std::span<const std::vector<std::uint32_t>>(scratch.users.data(), count), *hit, &gate);
    } catch (const std::exception&) {
        outcome = DrawRecords::Counter::MissFetchError;
    }
    if (outcome != DrawRecords::Counter::Hits) {
        miss = FetchMiss(outcome, gate);
        return nullptr;
    }
    hit->record = draw->record;
    return hit;
}

void Driver::finishStagePlanDraw(const LeanDrawJob& job) {
    using StagePlans::Counter;
    const auto& anchor = leanDraws.AnchorAt(job.anchor);
    const bool buildable = (job.recordFlags & DrawRecordFlags::BuildWanted) != 0;
    std::shared_ptr<DrawRecordHit> hit;
    auto miss = Counter::MissNoPlans;
    if (job.plans != nullptr) hit = lookupStagePlans(job, miss);
    if (hit != nullptr) {
        StagePlans::Add(Counter::Hits);
        if (hit->inputsMoved) StagePlans::Add(Counter::InputRefetches);
        if (hit->relocated) StagePlans::Add(Counter::HitsRelocated);
        if (hit->baseValueMoved) StagePlans::Add(Counter::HitsBaseValue);
        if (hit->buffersPatched) StagePlans::Add(Counter::HitsBuffersPatched);
        if ((job.recordFlags & DrawRecordFlags::VerifyDue) == 0) {
            commitRecordDraw(job, anchor, std::move(hit));
            return;
        }
    }
    DrawRecordView answer;
    if (hit != nullptr) {
        StagePlans::Add(Counter::HitsOnTodaysPath);
        answer.Capture(hit->stages, hit->parameters);
        answer.reads = hit->reads;
        answer.writes = hit->record->writes;
        answer.contentKey = hit->record->contentKey;
        answer.fragmentOutputs = hit->record->fragmentOutputs;
        if (hit->record->plan->addressed) answer.CaptureSnapshots(hit->snapshots);
        answer.fetched = std::move(hit->fetched);
        if (DrawRecords::CorruptingAnswers()) answer.parameters[3] ^= 1u;
    } else if (buildable) {
        StagePlans::Add(miss);
    }
    const bool build = hit == nullptr && buildable && BuildsOn(miss);
    if (hit == nullptr && !build) {
        finishLeanDraw(job, anchor);
        return;
    }
    struct SinkStorage {};
    auto& sink = HostThreadLocal<DrawRecordSink, SinkStorage>();
    sink.Reset();
    sink.plan = build;
    sink.verify = hit != nullptr;
    struct SinkScope {
        ~SinkScope() { DrawRecordSinkSlot::Set(nullptr); }
    } scope;
    DrawRecordSinkSlot::Set(&sink);
    finishLeanDraw(job, anchor);
    DrawRecordSinkSlot::Set(nullptr);
    if (hit != nullptr) {
        verifyStagePlanHit(job, *hit->record, answer, sink);
        return;
    }
    settleStagePlans(job, sink);
}

void Driver::verifyStagePlanHit(const LeanDrawJob& job, const DrawRecord& record, const DrawRecordView& answer, const DrawRecordSink& sink) {
    StagePlans::NoteChecked(DrawRecords::Item::Fetch);
    if (!compareRecordAnswer(job, record, answer, sink, &StagePlans::NoteDifference, &StagePlans::NoteAllowed)) return;
    auto demote = std::make_shared<StagePlanAnswer>();
    demote->shape = job.shape->shared_from_this();
    demote->generation = job.recordGeneration;
    demote->deviceSerial = job.deviceSerial;
    demote->demote = true;
    DrawFinisher::AnswerLean(std::move(demote));
}

void Driver::settleStagePlans(const LeanDrawJob& job, const DrawRecordSink& sink) {
    using StagePlans::Counter;
    if (!sink.queued || sink.contentKey.empty()) {
        StagePlans::Add(Counter::BuildsNotQueued);
        return;
    }
    if (sink.built == nullptr) {
        StagePlans::Add(Refusal(sink.refusal));
        return;
    }
    if (Graphics::RuntimeSwitches::Generation() != job.recordGeneration) {
        StagePlans::Add(Counter::BuildsGenerationMoved);
        return;
    }
    auto answer = std::make_shared<StagePlanAnswer>();
    answer->shape = job.shape->shared_from_this();
    answer->generation = job.recordGeneration;
    answer->deviceSerial = job.deviceSerial;
    answer->stages.reserve(sink.built->stages.size());
    for (const auto& stage : sink.built->stages) {
        auto plan = MakeStagePlan(stage);
        if (plan == nullptr) {
            StagePlans::Add(Counter::RefusedLayout);
            return;
        }
        answer->stages.push_back(std::move(plan));
    }
    DrawRecordKey key{};
    key.shape = job.shapeKey;
    answer->record = DrawRecord::Build(key, answer->shape, job.deviceSerial, job.recordGeneration, Graphics::Recorder::Presents(), sink);
    StagePlans::Add(Counter::Builds);
    if (!DrawFinisher::AnswerLean(std::move(answer))) {
        StagePlans::Add(Counter::BuildsDropped);
        StagePlans::Add(Counter::DroppedUnanswered);
    }
}

void Driver::installStagePlans(const LeanDrawJob& job, const StagePlanAnswer& answer) {
    using StagePlans::Counter;
    auto& lean = leanDraws;
    const auto front = DrawFinisher::Get().Front();
    const auto dropped = [&](Counter reason) {
        if (answer.demote) return;
        StagePlans::AddOnWorker(Counter::BuildsDropped);
        StagePlans::AddOnWorker(reason);
    };
    if (job.recordGeneration != lean.stamp || answer.shape == nullptr || answer.generation != lean.stamp) {
        dropped(Counter::DroppedStale);
        return;
    }
    auto* record = lean.Find(answer.shape->shape, answer.deviceSerial);
    if (record == nullptr || record->built != answer.shape) {
        dropped(Counter::DroppedShape);
        return;
    }
    if (answer.demote) {
        lean.RetirePlans(*record, front);
        record->plansRefused = true;
        StagePlans::AddOnWorker(Counter::Demotions);
        return;
    }
    if (record->plansRefused) {
        dropped(Counter::DroppedRefused);
        return;
    }
    StagePlanMerge merge;
    auto merged = MergeStagePlans(record->plans.get(), answer, merge);
    if (merged == nullptr) {
        dropped(Counter::DroppedStale);
        return;
    }
    if (!lean.InstallPlans(*record, std::move(merged), front, Graphics::Recorder::Presents())) {
        dropped(Counter::DroppedOverCap);
        return;
    }
    StagePlans::AddOnWorker(Counter::StagePlansAdded, merge.added);
    StagePlans::AddOnWorker(Counter::StagePlansShared, merge.shared);
    StagePlans::AddOnWorker(Counter::StagePlansReplaced, merge.replaced);
    StagePlans::AddOnWorker(Counter::StagePlansEvicted, merge.evicted);
    StagePlans::AddOnWorker(Counter::LayoutsAdded, merge.layouts);
    StagePlans::AddOnWorker(Counter::DrawPlansAdded, merge.drawsAdded);
    StagePlans::AddOnWorker(Counter::DrawPlansKnown, merge.drawsKnown);
    StagePlans::AddOnWorker(Counter::DrawPlansEvicted, merge.drawsEvicted);
}

void Driver::retireStagePlans(const std::byte* payload, std::shared_ptr<const void>&& answer) {
    if (answer == nullptr) return;
    const auto& job = *std::launder(reinterpret_cast<const LeanDrawJob*>(payload));
    const auto reply = std::static_pointer_cast<const StagePlanAnswer>(std::move(answer));
    Get().installStagePlans(job, *reply);
}

}
