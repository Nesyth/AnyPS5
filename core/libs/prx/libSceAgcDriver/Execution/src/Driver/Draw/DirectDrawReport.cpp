#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DirectDrawReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace AgcDriver::DriverDetail::DirectDraws {

namespace {

using DrawRecords::Allowed;
using DrawRecords::Field;
using DrawRecords::Item;

constexpr std::size_t CounterCount = static_cast<std::size_t>(Counter::Count);
constexpr std::size_t StepCount = static_cast<std::size_t>(Step::Count);
constexpr std::size_t FieldCount = static_cast<std::size_t>(Field::Count);
constexpr std::size_t AllowedKinds = static_cast<std::size_t>(Allowed::Count);
constexpr std::uint64_t PrintedPerWindow = 8;
constexpr std::uint64_t BuildLinesPerReason = 16;
constexpr std::int64_t WindowMs = 10000;

constexpr std::array<const char*, CounterCount> CounterNames{
    "draws", "direct",
    "shape", "heap", "invocation", "sample retry", "walk", "torn reads", "tables", "poisoned", "no patch", "addressed", "side", "push constants", "unsettled", "views not current", "memory changed", "labels", "parameter overflow", "class new", "class refused", "verify",
    "classes built", "not queued", "writes differ", "verify differs", "build differs", "classes evicted", "stages built", "stages evicted", "invocations rebuilt", "verify checked", "verify differing",
    "commit hits", "first commits", "commit misses", "commits demoted", "proofs skipped", "proofs not skipped", "full proofs",
    "held by the record", "held by the pass", "first use", "new pass", "collect epoch", "proof serial", "pending stores", "depth surfaces", "target changed",
    "buffers prepared", "no plan", "classification", "not accessible", "over the upload limit", "size budget", "not settled", "blob bytes",
    "buffers used", "plan differs", "moved-range hazard", "unmoved-range hazard", "depth guards", "dropped", "buffers verified", "buffers differing", "buffer memory changed", "buffers differing unwatched",
    "inputs prepared", "no published recipe", "outside the shape", "not imported in place", "inputs not accessible", "inputs threw", "indices not settled",
    "inputs used", "inputs rechecked", "import changed", "commit shape", "refused in place", "inputs dropped", "inputs verified", "inputs differing",
    "repeats taken", "repeat blocks built", "repeats beginning a pass", "repeats refused after a side effect", "repeat proof misses", "repeat proofs skipped", "repeat unmoved tests skipped", "repeats verified", "repeats differing", "repeat claims refuted",
    "no block", "not simple", "verify due", "shape", "objects", "another batch", "pass", "collect epoch", "proof serial", "pending stores", "depth surfaces", "target changed", "pass access", "queued stores", "drawn linear targets", "no prepared buffers", "selection", "depth guards", "inputs", "changed by the inputs", "moved-range hazard", "unmoved-range hazard",
    "not recorded", "template", "stages", "pipeline", "targets", "not pushed", "sets", "buffers", "inputs", "keep", "pass", "access", "notes", "modes",
    "full commit ns", "full commits timed", "repeat ns", "repeats timed",
    "repeats with a written set", "repeats beside queued stores", "repeats under the shared hold",
    "not reusable", "heap stage", "lease", "address table", "copied writes", "fault checks", "space reuse", "GPU writes", "GPU copies", "depth copies",
    "multisampled", "multisampled depth", "not the recipe's", "linear", "fmask",
    "set repeat ns", "set repeats timed", "store repeat ns", "store repeats timed", "shared repeat ns", "shared repeats timed",
    "bindings placed", "bindings not placed", "finisher slices bound", "finisher slices bound with a written set", "blobs copied", "placed for another device", "unmoved-range hazard", "depth guards", "slices returned", "bytes not copied", "objects not allocated",
    "unmoved tests", "unmoved tests answered by stamp", "guard walks answered by the prepared test", "sets lent", "sets written", "bindings verified", "bindings differing", "stamp claims refuted",
    "today's pushed ns", "today's pushed timed", "today's written-set ns", "today's written-set timed", "placed pushed ns", "placed pushed timed", "placed written-set ns", "placed written-set timed", "keep ns", "keeps timed", "dynamic sets bound", "dynamic sets written", "dynamic sets refused",
    "dynamic pushed sets bound", "dynamic tuple hits", "written pushed ns", "written pushed timed", "written unpushed ns", "written unpushed timed", "dynamic pushed ns", "dynamic pushed timed", "dynamic unpushed ns", "dynamic unpushed timed",
    "refused after a difference", "no prepared buffers", "stencil references or mesh window", "heap pipeline", "heap stage or binding", "address stages", "update after bind", "addressed buffers", "fault descriptors", "set or write shape", "per-draw uniform buffer", "per-draw binding of another type", "binding flags", "no per-draw storage buffer", "over the device limit", "push pipeline held", "other set without per-draw storage buffers", "before a second set was written", "repeat route", "pipeline not built", "rebound template or selection",
    "refused after a difference", "no prepared buffers", "stencil references or mesh window", "heap pipeline", "heap stage or binding", "address stages", "update after bind", "addressed buffers", "fault descriptors", "set or write shape", "per-draw uniform buffer", "per-draw binding of another type", "binding flags", "no per-draw storage buffer", "over the device limit", "push pipeline held", "other set without per-draw storage buffers", "before a second set was written", "repeat route", "pipeline not built", "rebound template or selection", "fragments made", "no emit block", "objects gone", "another device", "modes", "stages", "pipeline", "selection", "thrown", "fragments spliced", "fragments replayed", "fragments with a written set", "other objects", "selection", "inputs", "bindings", "modes", "record refused", "verify", "fragment verify checked", "fragment verify differing", "fragment finisher ns", "fragment finisher timed", "spliced commit ns", "spliced commits timed", "today's commit ns", "today's commits timed", "express commits", "no express state", "shape", "objects", "template", "targets", "depth", "queued stores", "drawn linear", "buffers", "inputs", "moved hazard", "selection", "staging", "verify", "express verify checked", "express verify differing", "express commit ns", "express commits timed", "reproved", "targets refreshed", "captured", "not spliced", "objects", "template not proved", "template written", "forget serial", "serials", "sources", "views", "pending CPU writes", "staging", "shadowed regions", "access", "not lean", "multisampled", "refused", "function LDS", "FMASK", "capture not reached", "not reusable", "space reuse", "BDA", "heap stage", "lease", "copied writes", "GPU copies", "buffer writes", "storage writes", "fault buffer", "no prepared buffers", "stencil references", "mesh window", "not eligible", "refused", "modes", "new exact key", "same key and values", "same key, other values", "content new", "content seen before"
};
static_assert(static_cast<std::size_t>(Counter::BindingsLeftUnpushedRebound) - static_cast<std::size_t>(Counter::BindingsLeftUnpushedRefused) + 2 == static_cast<std::size_t>(Graphics::ShaderResources::DynamicLeave::Count), "every way an unpushed draw leaves the dynamic route needs a counter");
static_assert(static_cast<std::size_t>(Counter::BindingsLeftPushedRebound) - static_cast<std::size_t>(Counter::BindingsLeftPushedRefused) + 2 == static_cast<std::size_t>(Graphics::ShaderResources::DynamicLeave::Count), "every way a pushed draw leaves the dynamic route needs a counter");

constexpr std::array<const char*, StepCount> StepNames{"request", "invocation", "walk", "key", "side", "class", "checks", "buffers", "inputs", "hand-off"};

static_assert(CounterNames.back() != nullptr && StepNames.back() != nullptr, "every direct draw counter and step needs a name");
static_assert(static_cast<std::size_t>(Counter::FragmentsThrown) - static_cast<std::size_t>(Counter::FragmentsNoBlock) + 2 == static_cast<std::size_t>(DrawFragmentRefusal::Count), "every reason a finisher makes no fragment needs a counter");
static_assert(static_cast<std::size_t>(Counter::ExpressCaptureUnreached) - static_cast<std::size_t>(Counter::ExpressCaptured) + 2 == static_cast<std::size_t>(DrawExpressCapture::Count), "every express capture outcome needs a counter");
static_assert(static_cast<std::size_t>(Counter::ExpressVerify) - static_cast<std::size_t>(Counter::ExpressNoState) + 2 == static_cast<std::size_t>(DrawExpressMiss::Count), "every reason an express commit is not taken needs a counter");
static_assert(static_cast<std::size_t>(Counter::FragmentsDropVerify) - static_cast<std::size_t>(Counter::FragmentsDropObjects) + 2 == static_cast<std::size_t>(DrawFragmentDrop::Count), "every reason a committer drops a fragment needs a counter");
static_assert(static_cast<std::size_t>(Counter::RepeatUnmovedHazard) - static_cast<std::size_t>(Counter::RepeatNoBlock) + 1 == static_cast<std::size_t>(Graphics::RepeatRefusal::Count), "every repeat refusal needs a counter");
static_assert(static_cast<std::size_t>(Counter::RepeatClassModes) - static_cast<std::size_t>(Counter::RepeatClassNotRecorded) + 2 == static_cast<std::size_t>(DrawRepeatClass::Count), "every repeat class refusal needs a counter");
static_assert(static_cast<std::size_t>(Counter::RepeatDetailFmask) - static_cast<std::size_t>(Counter::RepeatDetailNotReusable) + 2 == static_cast<std::size_t>(DrawRepeatDetail::Count), "every repeat class condition needs a counter");

struct Block {
    std::array<std::atomic<std::uint64_t>, CounterCount> counters{};
    std::array<std::atomic<std::uint64_t>, StepCount> stepNanoseconds{};
    std::array<std::atomic<std::uint64_t>, StepCount> stepDraws{};
};

struct Registry {
    std::mutex mutex;
    std::vector<Block*> live;
    Block retired;
};

Registry& registry() {
    static auto* kept = new Registry;
    return *kept;
}

void bump(std::atomic<std::uint64_t>& value, std::uint64_t count) {
    value.store(value.load(std::memory_order_relaxed) + count, std::memory_order_relaxed);
}

void fold(Block& into, const Block& from) {
    for (std::size_t index = 0; index < CounterCount; ++index) bump(into.counters[index], from.counters[index].load(std::memory_order_relaxed));
    for (std::size_t index = 0; index < StepCount; ++index) {
        bump(into.stepNanoseconds[index], from.stepNanoseconds[index].load(std::memory_order_relaxed));
        bump(into.stepDraws[index], from.stepDraws[index].load(std::memory_order_relaxed));
    }
}

struct BlockHolder {
    Block* block = nullptr;

    ~BlockHolder() {
        if (block == nullptr) return;
        auto& kept = registry();
        std::lock_guard lock(kept.mutex);
        fold(kept.retired, *block);
        std::erase(kept.live, block);
        delete block;
    }
};

struct BlockStorage {};

Block& own() {
    auto& holder = HostThreadLocal<BlockHolder, BlockStorage>();
    if (holder.block != nullptr) return *holder.block;
    auto block = std::make_unique<Block>();
    auto& kept = registry();
    std::lock_guard lock(kept.mutex);
    kept.live.push_back(block.get());
    holder.block = block.release();
    return *holder.block;
}

template <typename Read>
std::uint64_t summed(const Read& read) {
    auto& kept = registry();
    std::lock_guard lock(kept.mutex);
    auto total = read(kept.retired);
    for (const auto* block : kept.live) total += read(*block);
    return total;
}

struct Window {
    std::array<std::uint64_t, CounterCount> counters{};
    std::array<std::uint64_t, StepCount> stepNanoseconds{};
    std::array<std::uint64_t, StepCount> stepDraws{};
};

Window totals() {
    Window now;
    auto& kept = registry();
    std::lock_guard lock(kept.mutex);
    const auto add = [&](const Block& block) {
        for (std::size_t index = 0; index < CounterCount; ++index) now.counters[index] += block.counters[index].load(std::memory_order_relaxed);
        for (std::size_t index = 0; index < StepCount; ++index) {
            now.stepNanoseconds[index] += block.stepNanoseconds[index].load(std::memory_order_relaxed);
            now.stepDraws[index] += block.stepDraws[index].load(std::memory_order_relaxed);
        }
    };
    add(kept.retired);
    for (const auto* block : kept.live) add(*block);
    return now;
}

std::array<std::atomic<std::uint64_t>, FieldCount> differences{};
std::array<std::atomic<std::uint64_t>, FieldCount> buildDifferences{};
std::array<std::atomic<std::uint64_t>, AllowedKinds> allowed{};
std::atomic<std::uint64_t> printed{0};
std::atomic<std::uint64_t> bufferPrinted{0};
std::atomic<std::uint64_t> repeatPrinted{0};
std::atomic<std::uint64_t> bindingsPrinted{0};
std::atomic<std::uint64_t> classesKnown{0};
std::atomic<std::uint64_t> classBytes{0};
std::atomic<std::uint64_t> stagesKnown{0};
std::mutex reportMutex;
std::int64_t reportedAt = 0;
Window reported;

bool Printed() {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    return stats;
}

std::string reasons(const std::array<std::uint64_t, CounterCount>& counts, Counter first, Counter last) {
    std::string text;
    for (auto index = static_cast<std::size_t>(first); index <= static_cast<std::size_t>(last); ++index) {
        if (counts[index] == 0) continue;
        text += text.empty() ? " (" : ", ";
        text += CounterNames[index];
        text += ' ';
        text += std::to_string(counts[index]);
    }
    return text.empty() ? text : text + ")";
}

}

std::uint64_t VerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_DIRECT_DRAWS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

std::uint64_t BufferVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_DIRECT_BUFFERS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

std::uint64_t InputVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_DIRECT_INPUTS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

std::uint64_t RepeatVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_DIRECT_REPEAT");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

void NoteRepeatDifference(std::string_view field, std::string_view detail) {
    if (repeatPrinted.fetch_add(1, std::memory_order_relaxed) < PrintedPerWindow) std::fprintf(stderr, "[direct-repeat-verify] %.*s differs: %.*s\n", static_cast<int>(field.size()), field.data(), static_cast<int>(detail.size()), detail.data());
}

void NoteBindingsDifference(std::string_view field, std::string_view detail) {
    if (bindingsPrinted.fetch_add(1, std::memory_order_relaxed) < PrintedPerWindow) std::fprintf(stderr, "[commit-bindings-verify] %.*s differs: %.*s\n", static_cast<int>(field.size()), field.data(), static_cast<int>(detail.size()), detail.data());
}

void Add(Counter counter, std::uint64_t count) {
    bump(own().counters[static_cast<std::size_t>(counter)], count);
}

void AddStep(Step step, std::uint64_t nanoseconds) {
    auto& block = own();
    bump(block.stepNanoseconds[static_cast<std::size_t>(step)], nanoseconds);
    bump(block.stepDraws[static_cast<std::size_t>(step)], 1);
}

void NoteClasses(std::size_t known, std::size_t bytes) {
    classesKnown.store(known, std::memory_order_relaxed);
    classBytes.store(bytes, std::memory_order_relaxed);
}

void NoteStages(std::size_t known) {
    stagesKnown.store(known, std::memory_order_relaxed);
}

void NoteDifference(Field field, std::string_view detail) {
    differences[static_cast<std::size_t>(field)].fetch_add(1, std::memory_order_relaxed);
    if (printed.fetch_add(1, std::memory_order_relaxed) < PrintedPerWindow) std::fprintf(stderr, "[direct-draws-verify] %s %s differs: %.*s\n", DrawRecords::ItemName(Item::Fetch), DrawRecords::FieldName(field), static_cast<int>(detail.size()), detail.data());
}

void NoteBuildDifference(Field field, std::string_view detail) {
    if (buildDifferences[static_cast<std::size_t>(field)].fetch_add(1, std::memory_order_relaxed) < BuildLinesPerReason) std::fprintf(stderr, "[direct-draws-build] %s %s differs: %.*s\n", DrawRecords::ItemName(Item::Fetch), DrawRecords::FieldName(field), static_cast<int>(detail.size()), detail.data());
}

void NoteBufferDifference(std::string_view what, std::string_view detail) {
    if (bufferPrinted.fetch_add(1, std::memory_order_relaxed) < PrintedPerWindow) std::fprintf(stderr, "[direct-buffers-verify] %.*s differs: %.*s\n", static_cast<int>(what.size()), what.data(), static_cast<int>(detail.size()), detail.data());
}

void NoteAllowed(Allowed kind) {
    allowed[static_cast<std::size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
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
    const auto current = totals();
    Window since;
    for (std::size_t index = 0; index < CounterCount; ++index) since.counters[index] = current.counters[index] - reported.counters[index];
    for (std::size_t index = 0; index < StepCount; ++index) {
        since.stepNanoseconds[index] = current.stepNanoseconds[index] - reported.stepNanoseconds[index];
        since.stepDraws[index] = current.stepDraws[index] - reported.stepDraws[index];
    }
    reported = current;
    printed.store(0, std::memory_order_relaxed);
    bufferPrinted.store(0, std::memory_order_relaxed);
    repeatPrinted.store(0, std::memory_order_relaxed);
    bindingsPrinted.store(0, std::memory_order_relaxed);
    const auto& counts = since.counters;
    const auto at = [&](Counter counter) { return counts[static_cast<std::size_t>(counter)]; };
    const auto draws = at(Counter::Draws);
    std::uint64_t fallbacks = 0;
    for (auto index = static_cast<std::size_t>(Counter::FallbackShape); index <= static_cast<std::size_t>(Counter::FallbackVerify); ++index) fallbacks += counts[index];
    std::uint64_t refused = 0;
    for (auto index = static_cast<std::size_t>(Counter::RefusedNotQueued); index <= static_cast<std::size_t>(Counter::RefusedDiffers); ++index) refused += counts[index];
    if (!Printed() || (draws == 0 && at(Counter::ClassesBuilt) == 0 && at(Counter::ClassesEvicted) == 0 && at(Counter::StagesBuilt) == 0 && at(Counter::StagesEvicted) == 0)) return;
    const auto perDraw = [&](Step step) {
        const auto index = static_cast<std::size_t>(step);
        return since.stepDraws[index] == 0 ? 0.0 : static_cast<double>(since.stepNanoseconds[index]) / static_cast<double>(since.stepDraws[index]) / 1000.0;
    };
    const auto direct = at(Counter::Direct);
    AgcDriver::ProfilePrint_nid_no_patch("[direct-draws] %llu draws: %llu direct (%.1f%%), %llu fallback%s; classes %llu known, %llu built, %llu refused%s, %llu evicted, %.1f KiB; stages %llu known, %llu built; invocations %llu rebuilt; verify %llu checked, %llu differing; us per direct draw: request %.2f, invocation %.2f, walk %.2f, key %.2f, side %.2f, class %.2f, checks %.2f, buffers %.2f, inputs %.2f, hand-off %.2f\n",
        static_cast<unsigned long long>(draws), static_cast<unsigned long long>(direct), draws != 0 ? 100.0 * static_cast<double>(direct) / static_cast<double>(draws) : 0.0,
        static_cast<unsigned long long>(fallbacks), reasons(counts, Counter::FallbackShape, Counter::FallbackVerify).c_str(),
        static_cast<unsigned long long>(classesKnown.load(std::memory_order_relaxed)), static_cast<unsigned long long>(at(Counter::ClassesBuilt)), static_cast<unsigned long long>(refused), reasons(counts, Counter::RefusedNotQueued, Counter::RefusedDiffers).c_str(),
        static_cast<unsigned long long>(at(Counter::ClassesEvicted)), static_cast<double>(classBytes.load(std::memory_order_relaxed)) / 1024.0,
        static_cast<unsigned long long>(stagesKnown.load(std::memory_order_relaxed)), static_cast<unsigned long long>(at(Counter::StagesBuilt)),
        static_cast<unsigned long long>(at(Counter::InvocationsRebuilt)), static_cast<unsigned long long>(at(Counter::VerifyChecked)), static_cast<unsigned long long>(at(Counter::VerifyDiffering)),
        perDraw(Step::Request), perDraw(Step::Invocation), perDraw(Step::Walk), perDraw(Step::Key), perDraw(Step::Side), perDraw(Step::Class), perDraw(Step::Checks), perDraw(Step::Buffers), perDraw(Step::Inputs), perDraw(Step::HandOff));
    const auto commits = at(Counter::CommitHits) + at(Counter::CommitFirst) + at(Counter::CommitMisses);
    if (commits != 0) {
        std::uint64_t refreshed = 0;
        for (auto index = static_cast<std::size_t>(Counter::HoldFirst); index <= static_cast<std::size_t>(Counter::HoldTarget); ++index) refreshed += counts[index];
        AgcDriver::ProfilePrint_nid_no_patch("[direct-commits] %llu commits: %llu from the record (%.1f%%), %llu first commits, %llu misses, %llu demoted; template proofs: %llu skipped clean, %llu not skipped, %llu in full; targets: %llu held by the record, %llu held by the pass, %llu refreshed%s; stages evicted %llu\n",
            static_cast<unsigned long long>(commits), static_cast<unsigned long long>(at(Counter::CommitHits)), 100.0 * static_cast<double>(at(Counter::CommitHits)) / static_cast<double>(commits),
            static_cast<unsigned long long>(at(Counter::CommitFirst)), static_cast<unsigned long long>(at(Counter::CommitMisses)), static_cast<unsigned long long>(at(Counter::CommitDemoted)),
            static_cast<unsigned long long>(at(Counter::ProofsSkipped)), static_cast<unsigned long long>(at(Counter::ProofsNotSkipped)), static_cast<unsigned long long>(at(Counter::ProofsFull)),
            static_cast<unsigned long long>(at(Counter::HoldOwn)), static_cast<unsigned long long>(at(Counter::HoldShared)), static_cast<unsigned long long>(refreshed), reasons(counts, Counter::HoldFirst, Counter::HoldTarget).c_str(),
            static_cast<unsigned long long>(at(Counter::StagesEvicted)));
    }
    const auto sum = [&](Counter first, Counter last) {
        std::uint64_t total = 0;
        for (auto index = static_cast<std::size_t>(first); index <= static_cast<std::size_t>(last); ++index) total += counts[index];
        return total;
    };
    if (sum(Counter::BuffersPrepared, Counter::BuffersUnwatched) != 0) {
        const auto prepared = at(Counter::BuffersPrepared);
        AgcDriver::ProfilePrint_nid_no_patch("[direct-commits] buffers: %llu hits prepared, %llu used, %llu not prepared%s, %llu not used%s; %.1f blob bytes per prepared hit; verify %llu checked, %llu differing, %llu memory changed, %llu differing in unwatched memory\n",
            static_cast<unsigned long long>(prepared), static_cast<unsigned long long>(at(Counter::BuffersUsed)),
            static_cast<unsigned long long>(sum(Counter::BuffersNoPlan, Counter::BuffersUnsettled)), reasons(counts, Counter::BuffersNoPlan, Counter::BuffersUnsettled).c_str(),
            static_cast<unsigned long long>(sum(Counter::BuffersPlanDiffers, Counter::BuffersDropped)), reasons(counts, Counter::BuffersPlanDiffers, Counter::BuffersDropped).c_str(),
            prepared != 0 ? static_cast<double>(at(Counter::BuffersBlobBytes)) / static_cast<double>(prepared) : 0.0,
            static_cast<unsigned long long>(at(Counter::BuffersVerifyChecked)), static_cast<unsigned long long>(at(Counter::BuffersVerifyDiffering)), static_cast<unsigned long long>(at(Counter::BuffersMemoryChanged)), static_cast<unsigned long long>(at(Counter::BuffersUnwatched)));
    }
    if (sum(Counter::InputsPrepared, Counter::InputsVerifyDiffering) != 0) {
        AgcDriver::ProfilePrint_nid_no_patch("[direct-commits] inputs: %llu hits prepared, %llu used (%llu after an import recheck), %llu not prepared%s, %llu not used%s; verify %llu checked, %llu differing\n",
            static_cast<unsigned long long>(at(Counter::InputsPrepared)), static_cast<unsigned long long>(at(Counter::InputsUsed)), static_cast<unsigned long long>(at(Counter::InputsRechecked)),
            static_cast<unsigned long long>(sum(Counter::InputsNoRecipe, Counter::InputsUnsettled)), reasons(counts, Counter::InputsNoRecipe, Counter::InputsUnsettled).c_str(),
            static_cast<unsigned long long>(sum(Counter::InputsImportChanged, Counter::InputsDropped)), reasons(counts, Counter::InputsImportChanged, Counter::InputsDropped).c_str(),
            static_cast<unsigned long long>(at(Counter::InputsVerifyChecked)), static_cast<unsigned long long>(at(Counter::InputsVerifyDiffering)));
    }
    if (sum(Counter::RepeatTaken, Counter::RepeatSharedTimed) != 0) {
        const auto perCommit = [&](Counter nanoseconds, Counter timed) { return at(timed) == 0 ? 0.0 : static_cast<double>(at(nanoseconds)) / static_cast<double>(at(timed)) / 1000.0; };
        AgcDriver::ProfilePrint_nid_no_patch("[direct-commits] repeats: %llu taken (%llu beginning a pass, %llu proof misses, %llu proofs skipped, %llu unmoved tests skipped, %llu with a written set, %llu beside queued stores, %llu under the shared hold), %llu not taken%s, not simple%s, by condition%s; %llu blocks built, %llu refused after a side effect; verify %llu checked, %llu differing, %llu claims refuted; us per commit: full %.2f, repeat %.2f (with a written set %.2f, beside queued stores %.2f, under the shared hold %.2f)\n",
            static_cast<unsigned long long>(at(Counter::RepeatTaken)), static_cast<unsigned long long>(at(Counter::RepeatPassBegun)), static_cast<unsigned long long>(at(Counter::RepeatProofMisses)), static_cast<unsigned long long>(at(Counter::RepeatProofsSkipped)), static_cast<unsigned long long>(at(Counter::RepeatUnmovedSkipped)),
            static_cast<unsigned long long>(at(Counter::RepeatTakenSets)), static_cast<unsigned long long>(at(Counter::RepeatTakenStores)), static_cast<unsigned long long>(at(Counter::RepeatTakenShared)),
            static_cast<unsigned long long>(sum(Counter::RepeatNoBlock, Counter::RepeatUnmovedHazard)), reasons(counts, Counter::RepeatNoBlock, Counter::RepeatUnmovedHazard).c_str(), reasons(counts, Counter::RepeatClassNotRecorded, Counter::RepeatClassModes).c_str(), reasons(counts, Counter::RepeatDetailNotReusable, Counter::RepeatDetailFmask).c_str(),
            static_cast<unsigned long long>(at(Counter::RepeatBlocks)), static_cast<unsigned long long>(at(Counter::RepeatLate)),
            static_cast<unsigned long long>(at(Counter::RepeatVerifyChecked)), static_cast<unsigned long long>(at(Counter::RepeatVerifyDiffering)), static_cast<unsigned long long>(at(Counter::RepeatRefuted)),
            perCommit(Counter::RepeatFullNanoseconds, Counter::RepeatFullTimed), perCommit(Counter::RepeatNanoseconds, Counter::RepeatTimed), perCommit(Counter::RepeatSetsNanoseconds, Counter::RepeatSetsTimed), perCommit(Counter::RepeatStoresNanoseconds, Counter::RepeatStoresTimed), perCommit(Counter::RepeatSharedNanoseconds, Counter::RepeatSharedTimed));
    }
    if (sum(Counter::BindingsPlaced, Counter::BindingsLeftPushedRebound) != 0) {
        static Graphics::ShaderResources::UploadRingCounts ringReported;
        const auto ring = Graphics::ShaderResources::UploadRingTotals();
        const auto perDraw = [&](Counter nanoseconds, Counter timed) { return at(timed) == 0 ? 0.0 : static_cast<double>(at(nanoseconds)) / static_cast<double>(at(timed)); };
        AgcDriver::ProfilePrint_nid_no_patch("[direct-commits] bindings: %llu placed on the finisher, %llu not placed; %llu finisher slices bound (%llu with a written set), %llu blobs copied, %llu not taken%s, %llu slices returned unbound; %llu bytes not copied, %llu objects not allocated; unmoved tests %llu, %llu answered by stamp; %llu guard walks answered by the prepared depth test; per-draw sets %llu lent, %llu written; verify %llu checked, %llu differing, %llu stamp claims refuted; ns per binding step: today pushed %.0f (%llu), today with a written set %.0f (%llu), placed pushed %.0f (%llu), placed with a written set %.0f (%llu); ns per keep %.0f (%llu); dynamic sets: %llu unpushed draws, %llu pushed draws, %llu tuple hits, %llu tuple misses (sets written), %llu draws refused, %llu templates over the device limit; ns per written-set step: pushed today %.0f (%llu), unpushed today %.0f (%llu), pushed dynamic %.0f (%llu), unpushed dynamic %.0f (%llu); upload ring: %llu chunks, %llu reused, %llu added, %llu overflow, %llu busy at most; left the dynamic route with a written set: %llu unpushed%s, %llu pushed%s\n",
            static_cast<unsigned long long>(at(Counter::BindingsPlaced)), static_cast<unsigned long long>(at(Counter::BindingsNotPlaced)),
            static_cast<unsigned long long>(at(Counter::BindingsTaken)), static_cast<unsigned long long>(at(Counter::BindingsTakenWritten)), static_cast<unsigned long long>(at(Counter::BindingsCopied)),
            static_cast<unsigned long long>(sum(Counter::BindingsOtherDevice, Counter::BindingsDepthGuards)), reasons(counts, Counter::BindingsOtherDevice, Counter::BindingsDepthGuards).c_str(), static_cast<unsigned long long>(at(Counter::BindingsReturned)),
            static_cast<unsigned long long>(at(Counter::BindingsBytes)), static_cast<unsigned long long>(at(Counter::BindingsObjects)),
            static_cast<unsigned long long>(at(Counter::BindingsTested)), static_cast<unsigned long long>(at(Counter::BindingsStamped)), static_cast<unsigned long long>(at(Counter::BindingsGuards)),
            static_cast<unsigned long long>(at(Counter::BindingsSetsLent)), static_cast<unsigned long long>(at(Counter::BindingsSetsWritten)),
            static_cast<unsigned long long>(at(Counter::BindingsVerifyChecked)), static_cast<unsigned long long>(at(Counter::BindingsVerifyDiffering)), static_cast<unsigned long long>(at(Counter::BindingsRefuted)),
            perDraw(Counter::BindingsTodayPushedNanoseconds, Counter::BindingsTodayPushedTimed), static_cast<unsigned long long>(at(Counter::BindingsTodayPushedTimed)),
            perDraw(Counter::BindingsTodayWrittenNanoseconds, Counter::BindingsTodayWrittenTimed), static_cast<unsigned long long>(at(Counter::BindingsTodayWrittenTimed)),
            perDraw(Counter::BindingsNewPushedNanoseconds, Counter::BindingsNewPushedTimed), static_cast<unsigned long long>(at(Counter::BindingsNewPushedTimed)),
            perDraw(Counter::BindingsNewWrittenNanoseconds, Counter::BindingsNewWrittenTimed), static_cast<unsigned long long>(at(Counter::BindingsNewWrittenTimed)),
            perDraw(Counter::BindingsKeepNanoseconds, Counter::BindingsKeepTimed), static_cast<unsigned long long>(at(Counter::BindingsKeepTimed)),
            static_cast<unsigned long long>(at(Counter::BindingsDynamicBound)), static_cast<unsigned long long>(at(Counter::BindingsDynamicPushedBound)), static_cast<unsigned long long>(at(Counter::BindingsDynamicHits)), static_cast<unsigned long long>(at(Counter::BindingsDynamicWritten)), static_cast<unsigned long long>(at(Counter::BindingsDynamicRefused)), static_cast<unsigned long long>(Graphics::ShaderResources::DynamicPlansRefused()),
            perDraw(Counter::BindingsWrittenPushedNanoseconds, Counter::BindingsWrittenPushedTimed), static_cast<unsigned long long>(at(Counter::BindingsWrittenPushedTimed)),
            perDraw(Counter::BindingsWrittenUnpushedNanoseconds, Counter::BindingsWrittenUnpushedTimed), static_cast<unsigned long long>(at(Counter::BindingsWrittenUnpushedTimed)),
            perDraw(Counter::BindingsDynamicPushedNanoseconds, Counter::BindingsDynamicPushedTimed), static_cast<unsigned long long>(at(Counter::BindingsDynamicPushedTimed)),
            perDraw(Counter::BindingsDynamicUnpushedNanoseconds, Counter::BindingsDynamicUnpushedTimed), static_cast<unsigned long long>(at(Counter::BindingsDynamicUnpushedTimed)),
            static_cast<unsigned long long>(ring.chunks), static_cast<unsigned long long>(ring.reused - ringReported.reused), static_cast<unsigned long long>(ring.added - ringReported.added), static_cast<unsigned long long>(ring.overflow - ringReported.overflow), static_cast<unsigned long long>(ring.inFlight),
            static_cast<unsigned long long>(sum(Counter::BindingsLeftUnpushedRefused, Counter::BindingsLeftUnpushedRebound)), reasons(counts, Counter::BindingsLeftUnpushedRefused, Counter::BindingsLeftUnpushedRebound).c_str(),
            static_cast<unsigned long long>(sum(Counter::BindingsLeftPushedRefused, Counter::BindingsLeftPushedRebound)), reasons(counts, Counter::BindingsLeftPushedRefused, Counter::BindingsLeftPushedRebound).c_str());
        ringReported = ring;
    }
    if (sum(Counter::FragmentsMade, Counter::FragmentsTodayTimed) != 0) {
        const auto perCommit = [&](Counter nanoseconds, Counter timed) { return at(timed) == 0 ? 0.0 : static_cast<double>(at(nanoseconds)) / static_cast<double>(at(timed)); };
        AgcDriver::ProfilePrint_nid_no_patch("[direct-commits] fragments: %llu made on the finisher (%llu with a written set), %llu not made%s; %llu spliced, %llu replayed, %llu dropped%s; verify %llu checked, %llu differing; ns on the finisher per fragment %.0f (%llu); committer ns per commit: spliced %.0f (%llu), today's %.0f (%llu)\n",
            static_cast<unsigned long long>(at(Counter::FragmentsMade)), static_cast<unsigned long long>(at(Counter::FragmentsWritten)),
            static_cast<unsigned long long>(sum(Counter::FragmentsNoBlock, Counter::FragmentsThrown)), reasons(counts, Counter::FragmentsNoBlock, Counter::FragmentsThrown).c_str(),
            static_cast<unsigned long long>(at(Counter::FragmentsSpliced)), static_cast<unsigned long long>(at(Counter::FragmentsReplayed)),
            static_cast<unsigned long long>(sum(Counter::FragmentsDropObjects, Counter::FragmentsDropVerify)), reasons(counts, Counter::FragmentsDropObjects, Counter::FragmentsDropVerify).c_str(),
            static_cast<unsigned long long>(at(Counter::FragmentsVerifyChecked)), static_cast<unsigned long long>(at(Counter::FragmentsVerifyDiffering)),
            perCommit(Counter::FragmentsFinisherNanoseconds, Counter::FragmentsFinisherTimed), static_cast<unsigned long long>(at(Counter::FragmentsFinisherTimed)),
            perCommit(Counter::FragmentsSplicedNanoseconds, Counter::FragmentsSplicedTimed), static_cast<unsigned long long>(at(Counter::FragmentsSplicedTimed)),
            perCommit(Counter::FragmentsTodayNanoseconds, Counter::FragmentsTodayTimed), static_cast<unsigned long long>(at(Counter::FragmentsTodayTimed)));
    }
    if (sum(Counter::ResolveCensusNew, Counter::ResolveCensusChanged) != 0) {
        AgcDriver::ProfilePrint_nid_no_patch("[direct-draws] resolve census: %llu resolves%s; by shape, draw parameters and read content without addresses:%s\n", static_cast<unsigned long long>(sum(Counter::ResolveCensusNew, Counter::ResolveCensusChanged)), reasons(counts, Counter::ResolveCensusNew, Counter::ResolveCensusChanged).c_str(), reasons(counts, Counter::ResolveCensusContentNew, Counter::ResolveCensusContentSeen).c_str());
    }
    if (sum(Counter::FragmentsBlockNotReusable, Counter::FragmentsBlockModes) != 0) {
        AgcDriver::ProfilePrint_nid_no_patch("[direct-commits] fragment blocks not published: %llu commits%s\n", static_cast<unsigned long long>(sum(Counter::FragmentsBlockNotReusable, Counter::FragmentsBlockModes)), reasons(counts, Counter::FragmentsBlockNotReusable, Counter::FragmentsBlockModes).c_str());
    }
    if (sum(Counter::ExpressTaken, Counter::ExpressCaptureUnreached) != 0) {
        const auto perCommit = [&](Counter nanoseconds, Counter timed) { return at(timed) == 0 ? 0.0 : static_cast<double>(at(nanoseconds)) / static_cast<double>(at(timed)); };
        AgcDriver::ProfilePrint_nid_no_patch("[direct-commits] express: %llu taken, %llu not taken%s; verify %llu checked, %llu differing; ns per express commit %.0f (%llu); %llu reproved, %llu targets refreshed; states %llu captured, %llu not%s\n",
            static_cast<unsigned long long>(at(Counter::ExpressTaken)), static_cast<unsigned long long>(sum(Counter::ExpressNoState, Counter::ExpressVerify)), reasons(counts, Counter::ExpressNoState, Counter::ExpressVerify).c_str(),
            static_cast<unsigned long long>(at(Counter::ExpressVerifyChecked)), static_cast<unsigned long long>(at(Counter::ExpressVerifyDiffering)),
            perCommit(Counter::ExpressNanoseconds, Counter::ExpressTimed), static_cast<unsigned long long>(at(Counter::ExpressTimed)), static_cast<unsigned long long>(at(Counter::ExpressReproved)), static_cast<unsigned long long>(at(Counter::ExpressRefreshed)),
            static_cast<unsigned long long>(at(Counter::ExpressCaptured)), static_cast<unsigned long long>(sum(Counter::ExpressCaptureNotSpliced, Counter::ExpressCaptureUnreached)), reasons(counts, Counter::ExpressCaptureNotSpliced, Counter::ExpressCaptureUnreached).c_str());
    }
    if (commits == 0) return;
    static std::array<std::uint64_t, static_cast<std::size_t>(Graphics::RecipeRefusal::Count)> refusalsReported{};
    const auto refusals = Graphics::RecipeRefusals();
    std::string unbuilt;
    for (std::size_t index = 0; index < refusals.size(); ++index) {
        if (refusals[index] == refusalsReported[index]) continue;
        unbuilt += unbuilt.empty() ? "" : ", ";
        unbuilt += Graphics::RecipeRefusalNames[index];
        unbuilt += ' ';
        unbuilt += std::to_string(refusals[index] - refusalsReported[index]);
    }
    refusalsReported = refusals;
    if (!unbuilt.empty()) AgcDriver::ProfilePrint_nid_no_patch("[direct-commits] full draws that built no recipe: %s\n", unbuilt.c_str());
}

const char* CounterName(Counter counter) {
    return CounterNames[static_cast<std::size_t>(counter)];
}

std::uint64_t Total(Counter counter) {
    const auto index = static_cast<std::size_t>(counter);
    return summed([&](const Block& block) { return block.counters[index].load(std::memory_order_relaxed); });
}

std::uint64_t Differences(Field field) {
    return differences[static_cast<std::size_t>(field)].load(std::memory_order_relaxed);
}

std::uint64_t BuildDifferences(Field field) {
    return buildDifferences[static_cast<std::size_t>(field)].load(std::memory_order_relaxed);
}

std::uint64_t AllowedCount(Allowed kind) {
    return allowed[static_cast<std::size_t>(kind)].load(std::memory_order_relaxed);
}

}
