#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace AgcDriver::Graphics {

namespace {

constexpr std::size_t SwitchCount = static_cast<std::size_t>(RuntimeSwitch::Count);

constexpr std::array<const char*, SwitchCount> SwitchNames{
    "APS5_NO_BARRIER_BATCH",
    "APS5_NO_TEXTURE_PROOF_STAMPS",
    "APS5_NO_RANGE_PROOFS",
    "APS5_NO_LEAN_RUNS",
    "APS5_NO_PASS_ACCESS_MEMO",
    "APS5_NO_RANGED_DISPATCH_BREAKS",
    "APS5_NO_STAGE_DATA_SIDE_CARS",
    "APS5_NO_DATA_RECIPES",
    "APS5_NO_HEAP_RECIPES",
    "APS5_NO_INTERVAL_COLLECT_MEMO",
    "APS5_NO_INPUT_MEMO",
    "APS5_NO_PASS_ATTACHMENT_MEMO",
    "APS5_NO_GUARD_SPANS",
    "APS5_NO_HTILE_FIXUP_CARRY",
    "APS5_XP_WALKS_FREE",
    "APS5_NO_ARM_RESIDENT",
    "APS5_PIPELINE_DISPATCH",
    "APS5_NO_HEAP_SET_SKIP",
    "APS5_LARGE_BATCHES",
    "APS5_REPLAY_HEAP_STAGE",
    "APS5_XP_REPLAY",
    "APS5_XP_PROOF_AS_IS",
    "APS5_XP_STAGES_AS_IS",
    "APS5_REARM_QUIET",
    "APS5_CLEAN_CURRENCY",
    "APS5_ARM_WIDE",
    "APS5_ARM_SPANS",
    "APS5_REPLAY_BIND",
    "APS5_XP_UPLOAD_NO_SYNC",
    "APS5_XP_NO_EAGER_STORE",
    "APS5_ASYNC_SUBMIT",
    "APS5_PRECISE_LOOKUP_DRAIN",
    "APS5_SECOND_FINISHER",
    "APS5_THIRD_FINISHER",
    "APS5_LOCKLESS_HAZARDS",
    "APS5_XP_HALF_DRAWS",
    "APS5_XP_HALF_WORKER_DRAWS",
    "APS5_SPIN_HANDOFF",
    "APS5_BESIDE_SAMPLE_DUMPS",
    "APS5_DEEP_QUEUES",
    "APS5_SPECULATIVE_LOOKUP",
    "APS5_HOST_LARGE_CACHE",
    "APS5_PIPELINED_GPU_INDIRECT",
    "APS5_HANDOFF_GPU_INDIRECT",
    "APS5_INPUT_SCRATCH",
    "APS5_CARRY_VALIDATION",
    "APS5_SINGLE_INDEX_CHECK",
    "APS5_OWN_DRAW_SETS",
    "APS5_GUARD_RANGE_MEMO",
    "APS5_MOVED_LARGE_UPLOADS",
    "APS5_UNLOCKED_SNAPSHOT_RECHECK",
    "APS5_KEEP_BYTES_ONCE",
    "APS5_MSAA_PASS_MEMO",
    "APS5_GPU_SINGLE_KEYS",
    "APS5_LEAN_REMARK",
    "APS5_KEEP_FAST_PATH",
    "APS5_LAZY_PASS_READS",
    "APS5_RECYCLE_COMMAND_BUFFERS",
    "APS5_PUSH_MEMCPY",
    "APS5_ARM_FLIP_RESET",
    "APS5_RESIDENT_FIRST_COLLECT",
    "APS5_SORTED_HOST_REGIONS",
    "APS5_HOST_PREFETCH",
    "APS5_SKIP_HTILE_KEY_PRESYNC",
    "APS5_XP_KEEP_DRAWS",
    "APS5_XP_STREAM_SEGMENTS",
    "APS5_LEAN_FRONT",
    "APS5_DRAW_RECORDS",
    "APS5_STAGE_PLANS",
    "APS5_DIRECT_DRAWS",
    "APS5_DIRECT_CLEAN_PROOFS",
    "APS5_DIRECT_PASS_TARGETS",
    "APS5_CLEAN_COLLECTS",
    "APS5_DIRECT_BUFFERS",
    "APS5_PIPELINE_DISPATCH_MISSES",
    "APS5_XP_COMMIT_FLOOR",
    "APS5_MSAA_RECORDS",
    "APS5_XP_INPUTS_AS_IS",
    "APS5_XP_STAMPS_BY_EPOCH",
    "APS5_MSAA_RECORD_HOLD",
    "APS5_TEMPLATE_TOUCH",
    "APS5_XP_BINDINGS_AS_IS",
    "APS5_NO_PLAN_CLEAN_PROOFS",
    "APS5_DEFERRED_ENCODE",
    "APS5_NO_OWNED_DRAW_INPUTS",
    "APS5_DIRECT_INPUTS",
    "APS5_DIRECT_RETRY",
    "APS5_ITEMS_BY_TURN",
    "APS5_XP_NO_COMMITS",
    "APS5_FOURTH_FINISHER",
    "APS5_FIFTH_FINISHER",
    "APS5_LEAN_LIST_BATCH",
    "APS5_LEAN_LIST_DIRECT",
    "APS5_DIRECT_REPEAT",
    "APS5_CLEAN_RESIDENT_VIEWS",
    "APS5_CLEAN_PENDING_RANGES",
    "APS5_REFUSED_COLLECT_MEMO",
    "APS5_WORKER_CORE",
    "APS5_DIRECT_REPEAT_SETS",
    "APS5_DIRECT_REPEAT_STORES",
    "APS5_DIRECT_REPEAT_SHARED",
    "APS5_FINISHER_DEPOSIT",
    "APS5_COMMIT_BINDINGS",
    "APS5_COMMIT_BINDINGS_STAMPS",
    "APS5_COMMIT_BINDINGS_SETS",
    "APS5_COMMIT_BINDINGS_DYNAMIC",
    "APS5_COMMIT_FRAGMENTS",
    "APS5_EXPRESS_COMMITS",
    "APS5_LABEL_ORDERED_STORES",
    "APS5_DEEPER_RING",
    "APS5_DISPATCH_BESIDE",
    "APS5_FAULT_FRAGMENTS",
    "APS5_STAGE_MEMO",
    "APS5_KEEP_FAULT_SLOTS",
    "APS5_SIBLING_FINISHERS",
    "APS5_FAST_SETTLE",
    "APS5_XP_SLOW_FINISHER",
    "APS5_XP_SLOW_COMMITTER",
    "APS5_XP_SLOW_WORKER",
    "APS5_XP_EXPRESS_FLOOR",
    "APS5_LEAN_EXPRESS",
    "APS5_XP_APPEND_COMMITS",
    "APS5_PRECISE_SETTLE",
    "APS5_APPEND_COMMITS",
    "APS5_XP_NO_BARRIERS",
    "APS5_DRAW_RUNS",
    "APS5_LEAN_PREPARE",
    "APS5_POOLED_HITS",
    "APS5_EXPRESS_TARGET_UPLOADS",
    "APS5_ALIASES_AT_PREPARE",
    "APS5_STAGE_MEMO_ADMIT_FIRST",
    "APS5_XP_NO_OCCLUSION",
    "APS5_LABEL_READBACK_MEMO",
    "APS5_STAGE_MEMO_NO_INDEX",
    "APS5_PASS_BARRIERS_MERGED",
    "APS5_EXPRESS_PASS_STAMPS",
    "APS5_COMMIT_STAMPS",
    "APS5_PASS_READS_ONCE",
    "APS5_REAP_THROTTLE",
    "APS5_OCCLUSION_WINDOWS",
    "APS5_STAGE_MEMO_ADMIT_THIRD",
    "APS5_TURN_WAITS_BY_JOB",
    "APS5_STAGE_MEMO_FRONT",
    "APS5_EXPRESS_KEEP_STATE",
    "APS5_RUN_FOLLOW",
    "APS5_STAGE_MEMO_PATCH_VERTEX",
    "APS5_STAGE_MEMO_PATCH_BUFFERS",
    "APS5_STAGE_MEMO_PATCH_DATA",
    "APS5_XP_ATTACHMENT_LAYOUTS",
};
static_assert(SwitchNames.back() != nullptr, "every runtime switch needs a name");

struct SwitchValues {
    std::array<std::atomic<bool>, SwitchCount> set{};
    std::atomic<std::uint64_t> generation{1};
    SwitchValues() {
        const bool lateMsOff = std::getenv("APS5_LATE_MS") == nullptr;
        const std::array<bool, SwitchCount> environment{
            std::getenv("APS5_NO_BARRIER_BATCH") != nullptr,
            lateMsOff || std::getenv("APS5_NO_TEXTURE_PROOF_STAMPS") != nullptr,
            lateMsOff || std::getenv("APS5_NO_RANGE_PROOFS") != nullptr,
            lateMsOff || std::getenv("APS5_NO_LEAN_RUNS") != nullptr,
            lateMsOff || std::getenv("APS5_NO_PASS_ACCESS_MEMO") != nullptr,
            lateMsOff || std::getenv("APS5_NO_RANGED_DISPATCH_BREAKS") != nullptr,
            lateMsOff || std::getenv("APS5_NO_STAGE_DATA_SIDE_CARS") != nullptr,
            lateMsOff || std::getenv("APS5_NO_DATA_RECIPES") != nullptr,
            std::getenv("APS5_NO_HEAP_RECIPES") != nullptr,
            lateMsOff || std::getenv("APS5_NO_INTERVAL_COLLECT_MEMO") != nullptr,
            lateMsOff || std::getenv("APS5_NO_INPUT_MEMO") != nullptr,
            std::getenv("APS5_NO_PASS_ATTACHMENT_MEMO") != nullptr,
            std::getenv("APS5_NO_GUARD_SPANS") != nullptr,
            std::getenv("APS5_NO_HTILE_FIXUP_CARRY") != nullptr,
            std::getenv("APS5_XP_WALKS_FREE") != nullptr,
            std::getenv("APS5_NO_ARM_RESIDENT") != nullptr,
            [] {
                const char* text = std::getenv("APS5_PIPELINE_DISPATCH");
                return text != nullptr && text[0] != '\0' && text[0] != '0';
            }(),
            std::getenv("APS5_NO_HEAP_SET_SKIP") != nullptr,
            std::getenv("APS5_LARGE_BATCHES") != nullptr,
            [] {
                const char* text = std::getenv("APS5_REPLAY_HEAP_STAGE");
                return text != nullptr && text[0] != '\0' && text[0] != '0';
            }(),
            [] {
                const char* text = std::getenv("APS5_XP_REPLAY");
                return text != nullptr && text[0] != '\0' && text[0] != '0';
            }(),
            std::getenv("APS5_XP_PROOF_AS_IS") != nullptr,
            std::getenv("APS5_XP_STAGES_AS_IS") != nullptr,
            std::getenv("APS5_REARM_QUIET") != nullptr,
            std::getenv("APS5_CLEAN_CURRENCY") != nullptr,
            std::getenv("APS5_ARM_WIDE") != nullptr,
            std::getenv("APS5_ARM_SPANS") != nullptr,
            std::getenv("APS5_REPLAY_BIND") != nullptr,
            std::getenv("APS5_XP_UPLOAD_NO_SYNC") != nullptr,
            std::getenv("APS5_XP_NO_EAGER_STORE") != nullptr,
            std::getenv("APS5_ASYNC_SUBMIT") != nullptr,
            std::getenv("APS5_PRECISE_LOOKUP_DRAIN") != nullptr,
            std::getenv("APS5_SECOND_FINISHER") != nullptr,
            std::getenv("APS5_THIRD_FINISHER") != nullptr,
            std::getenv("APS5_LOCKLESS_HAZARDS") != nullptr,
            std::getenv("APS5_XP_HALF_DRAWS") != nullptr,
            std::getenv("APS5_XP_HALF_WORKER_DRAWS") != nullptr,
            std::getenv("APS5_SPIN_HANDOFF") != nullptr,
            std::getenv("APS5_BESIDE_SAMPLE_DUMPS") != nullptr,
            std::getenv("APS5_DEEP_QUEUES") != nullptr,
            std::getenv("APS5_SPECULATIVE_LOOKUP") != nullptr,
            std::getenv("APS5_HOST_LARGE_CACHE") != nullptr,
            std::getenv("APS5_PIPELINED_GPU_INDIRECT") != nullptr,
            std::getenv("APS5_HANDOFF_GPU_INDIRECT") != nullptr,
            std::getenv("APS5_INPUT_SCRATCH") != nullptr,
            std::getenv("APS5_CARRY_VALIDATION") != nullptr,
            std::getenv("APS5_SINGLE_INDEX_CHECK") != nullptr,
            std::getenv("APS5_OWN_DRAW_SETS") != nullptr,
            std::getenv("APS5_GUARD_RANGE_MEMO") != nullptr,
            std::getenv("APS5_MOVED_LARGE_UPLOADS") != nullptr,
            std::getenv("APS5_UNLOCKED_SNAPSHOT_RECHECK") != nullptr,
            std::getenv("APS5_KEEP_BYTES_ONCE") != nullptr,
            std::getenv("APS5_MSAA_PASS_MEMO") != nullptr,
            std::getenv("APS5_GPU_SINGLE_KEYS") != nullptr,
            std::getenv("APS5_LEAN_REMARK") != nullptr,
            std::getenv("APS5_KEEP_FAST_PATH") != nullptr,
            std::getenv("APS5_LAZY_PASS_READS") != nullptr,
            std::getenv("APS5_RECYCLE_COMMAND_BUFFERS") != nullptr,
            std::getenv("APS5_PUSH_MEMCPY") != nullptr,
            std::getenv("APS5_ARM_FLIP_RESET") != nullptr,
            std::getenv("APS5_RESIDENT_FIRST_COLLECT") != nullptr,
            std::getenv("APS5_SORTED_HOST_REGIONS") != nullptr,
            std::getenv("APS5_HOST_PREFETCH") != nullptr,
            std::getenv("APS5_SKIP_HTILE_KEY_PRESYNC") != nullptr,
            std::getenv("APS5_XP_KEEP_DRAWS") != nullptr,
            std::getenv("APS5_XP_STREAM_SEGMENTS") != nullptr,
            std::getenv("APS5_LEAN_FRONT") != nullptr,
            std::getenv("APS5_DRAW_RECORDS") != nullptr,
            std::getenv("APS5_STAGE_PLANS") != nullptr,
            std::getenv("APS5_DIRECT_DRAWS") != nullptr,
            std::getenv("APS5_DIRECT_CLEAN_PROOFS") != nullptr,
            std::getenv("APS5_DIRECT_PASS_TARGETS") != nullptr,
            std::getenv("APS5_CLEAN_COLLECTS") != nullptr,
            std::getenv("APS5_DIRECT_BUFFERS") != nullptr,
            std::getenv("APS5_PIPELINE_DISPATCH_MISSES") != nullptr,
            std::getenv("APS5_XP_COMMIT_FLOOR") != nullptr,
            std::getenv("APS5_MSAA_RECORDS") != nullptr,
            std::getenv("APS5_XP_INPUTS_AS_IS") != nullptr,
            std::getenv("APS5_XP_STAMPS_BY_EPOCH") != nullptr,
            std::getenv("APS5_MSAA_RECORD_HOLD") != nullptr,
            std::getenv("APS5_TEMPLATE_TOUCH") != nullptr,
            std::getenv("APS5_XP_BINDINGS_AS_IS") != nullptr,
            std::getenv("APS5_NO_PLAN_CLEAN_PROOFS") != nullptr,
            std::getenv("APS5_DEFERRED_ENCODE") != nullptr,
            std::getenv("APS5_NO_OWNED_DRAW_INPUTS") != nullptr,
            std::getenv("APS5_DIRECT_INPUTS") != nullptr,
            std::getenv("APS5_DIRECT_RETRY") != nullptr,
            std::getenv("APS5_ITEMS_BY_TURN") != nullptr,
            std::getenv("APS5_XP_NO_COMMITS") != nullptr,
            std::getenv("APS5_FOURTH_FINISHER") != nullptr,
            std::getenv("APS5_FIFTH_FINISHER") != nullptr,
            std::getenv("APS5_LEAN_LIST_BATCH") != nullptr,
            std::getenv("APS5_LEAN_LIST_DIRECT") != nullptr,
            std::getenv("APS5_DIRECT_REPEAT") != nullptr,
            std::getenv("APS5_CLEAN_RESIDENT_VIEWS") != nullptr,
            std::getenv("APS5_CLEAN_PENDING_RANGES") != nullptr,
            std::getenv("APS5_REFUSED_COLLECT_MEMO") != nullptr,
            std::getenv("APS5_WORKER_CORE") != nullptr,
            std::getenv("APS5_DIRECT_REPEAT_SETS") != nullptr,
            std::getenv("APS5_DIRECT_REPEAT_STORES") != nullptr,
            std::getenv("APS5_DIRECT_REPEAT_SHARED") != nullptr,
            std::getenv("APS5_FINISHER_DEPOSIT") != nullptr,
            std::getenv("APS5_COMMIT_BINDINGS") != nullptr,
            std::getenv("APS5_COMMIT_BINDINGS_STAMPS") != nullptr,
            std::getenv("APS5_COMMIT_BINDINGS_SETS") != nullptr,
            std::getenv("APS5_COMMIT_BINDINGS_DYNAMIC") != nullptr,
            std::getenv("APS5_COMMIT_FRAGMENTS") != nullptr,
            std::getenv("APS5_EXPRESS_COMMITS") != nullptr,
            std::getenv("APS5_LABEL_ORDERED_STORES") != nullptr,
            std::getenv("APS5_DEEPER_RING") != nullptr,
            std::getenv("APS5_DISPATCH_BESIDE") != nullptr,
            std::getenv("APS5_FAULT_FRAGMENTS") != nullptr,
            std::getenv("APS5_STAGE_MEMO") != nullptr,
            std::getenv("APS5_KEEP_FAULT_SLOTS") != nullptr,
            std::getenv("APS5_SIBLING_FINISHERS") != nullptr,
            std::getenv("APS5_FAST_SETTLE") != nullptr,
            std::getenv("APS5_XP_SLOW_FINISHER") != nullptr,
            std::getenv("APS5_XP_SLOW_COMMITTER") != nullptr,
            std::getenv("APS5_XP_SLOW_WORKER") != nullptr,
            std::getenv("APS5_XP_EXPRESS_FLOOR") != nullptr,
            std::getenv("APS5_LEAN_EXPRESS") != nullptr,
            std::getenv("APS5_XP_APPEND_COMMITS") != nullptr,
            std::getenv("APS5_PRECISE_SETTLE") != nullptr,
            std::getenv("APS5_APPEND_COMMITS") != nullptr,
            std::getenv("APS5_XP_NO_BARRIERS") != nullptr,
            std::getenv("APS5_DRAW_RUNS") != nullptr,
            std::getenv("APS5_LEAN_PREPARE") != nullptr,
            std::getenv("APS5_POOLED_HITS") != nullptr,
            std::getenv("APS5_EXPRESS_TARGET_UPLOADS") != nullptr,
            std::getenv("APS5_ALIASES_AT_PREPARE") != nullptr,
            std::getenv("APS5_STAGE_MEMO_ADMIT_FIRST") != nullptr,
            std::getenv("APS5_XP_NO_OCCLUSION") != nullptr,
            std::getenv("APS5_LABEL_READBACK_MEMO") != nullptr,
            std::getenv("APS5_STAGE_MEMO_NO_INDEX") != nullptr,
            std::getenv("APS5_PASS_BARRIERS_MERGED") != nullptr,
            std::getenv("APS5_EXPRESS_PASS_STAMPS") != nullptr,
            std::getenv("APS5_COMMIT_STAMPS") != nullptr,
            std::getenv("APS5_PASS_READS_ONCE") != nullptr,
            std::getenv("APS5_REAP_THROTTLE") != nullptr,
            std::getenv("APS5_OCCLUSION_WINDOWS") != nullptr,
            std::getenv("APS5_STAGE_MEMO_ADMIT_THIRD") != nullptr,
            std::getenv("APS5_TURN_WAITS_BY_JOB") != nullptr,
            std::getenv("APS5_STAGE_MEMO_FRONT") != nullptr,
            std::getenv("APS5_EXPRESS_KEEP_STATE") != nullptr,
            std::getenv("APS5_RUN_FOLLOW") != nullptr,
            std::getenv("APS5_STAGE_MEMO_PATCH_VERTEX") != nullptr,
            std::getenv("APS5_STAGE_MEMO_PATCH_BUFFERS") != nullptr,
            std::getenv("APS5_STAGE_MEMO_PATCH_DATA") != nullptr,
            std::getenv("APS5_XP_ATTACHMENT_LAYOUTS") != nullptr,
        };
        for (std::size_t index = 0; index < SwitchCount; ++index) set[index].store(environment[index], std::memory_order_relaxed);
    }
};

SwitchValues& Values() {
    static SwitchValues values;
    return values;
}

std::string_view Trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r' || text.back() == '\n')) text.remove_suffix(1);
    return text;
}

double ParseSeconds(std::string_view text, const char* what) {
    const std::string copy(Trim(text));
    char* end = nullptr;
    const double value = std::strtod(copy.c_str(), &end);
    if (copy.empty() || end != copy.c_str() + copy.size() || !std::isfinite(value) || value < 0) throw std::invalid_argument(std::string(what) + ": '" + copy + "' is not a number of seconds");
    return value;
}

std::string FlippableList() {
    std::string list;
    for (const auto* name : SwitchNames) list += std::string(list.empty() ? "" : ", ") + name;
    return list;
}

}

namespace RuntimeSwitches {

const char* Name(RuntimeSwitch id) {
    return SwitchNames.at(static_cast<std::size_t>(id));
}

bool Set(RuntimeSwitch id) {
    return Values().set.at(static_cast<std::size_t>(id)).load(std::memory_order_relaxed);
}

void Assign(RuntimeSwitch id, bool set) {
    auto& values = Values();
    if (values.set.at(static_cast<std::size_t>(id)).exchange(set, std::memory_order_relaxed) != set) values.generation.fetch_add(1, std::memory_order_release);
}

std::uint64_t Generation() {
    return Values().generation.load(std::memory_order_acquire);
}

SwitchSnapshot Snapshot() {
    auto& values = Values();
    SwitchSnapshot snapshot;
    for (std::size_t index = 0; index < SwitchCount; ++index) {
        if (values.set[index].load(std::memory_order_relaxed)) snapshot.Assign(static_cast<RuntimeSwitch>(index), true);
    }
    return snapshot;
}

std::optional<RuntimeSwitch> Find(std::string_view name) {
    for (std::size_t index = 0; index < SwitchCount; ++index) {
        if (name == SwitchNames[index]) return static_cast<RuntimeSwitch>(index);
    }
    return std::nullopt;
}

}

AbSchedule::Config AbSchedule::Parse(std::string_view ab, std::string_view start, std::string_view settle) {
    Config config;
    ab = Trim(ab);
    const auto colon = ab.find(':');
    if (colon == std::string_view::npos) throw std::invalid_argument("APS5_AB: expected <seconds>:<switch>[+<switch>...], got '" + std::string(ab) + "'");
    config.phaseSeconds = ParseSeconds(ab.substr(0, colon), "APS5_AB phase length");
    if (config.phaseSeconds < 1) throw std::invalid_argument("APS5_AB: a phase must last at least 1 s");
    auto list = ab.substr(colon + 1);
    if (Trim(list).empty()) throw std::invalid_argument("APS5_AB: no switch listed after the phase length");
    for (;;) {
        const auto plus = list.find('+');
        const auto name = Trim(list.substr(0, plus));
        if (name.empty()) throw std::invalid_argument("APS5_AB: an empty switch name in '" + std::string(ab) + "'");
        const auto found = RuntimeSwitches::Find(name);
        if (!found.has_value()) throw std::invalid_argument("APS5_AB: " + std::string(name) + " cannot change while the title runs; switches the A/B can change: " + FlippableList());
        if (std::find(config.switches.begin(), config.switches.end(), *found) != config.switches.end()) throw std::invalid_argument("APS5_AB: " + std::string(name) + " is listed twice");
        config.switches.push_back(*found);
        if (plus == std::string_view::npos) break;
        list = list.substr(plus + 1);
    }
    start = Trim(start);
    if (!start.empty()) {
        if (start.back() == 's') {
            config.startSeconds = ParseSeconds(start.substr(0, start.size() - 1), "APS5_AB_START");
        } else {
            const std::string copy(start);
            char* end = nullptr;
            errno = 0;
            const auto value = std::strtoull(copy.c_str(), &end, 10);
            if (end != copy.c_str() + copy.size() || errno == ERANGE || copy.front() < '0' || copy.front() > '9') throw std::invalid_argument("APS5_AB_START: '" + copy + "' is neither a present index nor <n>s");
            config.startPresent = value;
        }
    }
    if (!Trim(settle).empty()) config.settleSeconds = ParseSeconds(settle, "APS5_AB_SETTLE");
    if (config.settleSeconds >= config.phaseSeconds) throw std::invalid_argument("APS5_AB_SETTLE must be shorter than the phase");
    return config;
}

AbSchedule::AbSchedule(Config config) : config(std::move(config)) {}

AbSchedule::Step AbSchedule::AtPresent(std::uint64_t present, double seconds) {
    if (firstSeconds.has_value() && seconds < lastSeconds) {
        char text[160];
        std::snprintf(text, sizeof(text), "APS5_AB: present %llu at %.3f s comes before the previous present's %.3f s; every present must read one clock", static_cast<unsigned long long>(present), seconds, lastSeconds);
        throw std::logic_error(text);
    }
    lastSeconds = seconds;
    if (!firstSeconds.has_value()) firstSeconds = seconds;
    if (!started) {
        const bool due = config.startSeconds.has_value() ? seconds - *firstSeconds >= *config.startSeconds : present >= config.startPresent;
        if (!due) return {};
        started = true;
        settled = false;
        phase = 0;
        phaseStart = seconds;
        return {Event::Begin, 0, config.firstB};
    }
    if (seconds - phaseStart >= config.phaseSeconds) {
        ++phase;
        settled = false;
        phaseStart = seconds;
        return {Event::Begin, phase, ((phase & 1u) != 0) != config.firstB};
    }
    if (!settled && seconds - phaseStart >= config.settleSeconds) {
        settled = true;
        return {Event::Settled, phase, ((phase & 1u) != 0) != config.firstB};
    }
    return {};
}

namespace {

struct AbState {
    std::mutex mutex;
    std::optional<AbSchedule> schedule;
    std::array<bool, SwitchCount> environment{};
    std::chrono::steady_clock::time_point origin = std::chrono::steady_clock::now();
};

AbState& Ab() {
    static AbState* state = [] {
        auto* created = new AbState();
        const char* ab = std::getenv("APS5_AB");
        const char* start = std::getenv("APS5_AB_START");
        const char* settle = std::getenv("APS5_AB_SETTLE");
        if (ab != nullptr) {
            auto config = AbSchedule::Parse(ab, start != nullptr ? start : "", settle != nullptr ? settle : "");
            config.firstB = std::getenv("APS5_AB_FIRST_B") != nullptr;
            created->schedule.emplace(std::move(config));
            for (std::size_t index = 0; index < SwitchCount; ++index) created->environment[index] = RuntimeSwitches::Set(static_cast<RuntimeSwitch>(index));
        }
        return created;
    }();
    return *state;
}

std::string SwitchSettings(const AbState& state, bool b) {
    std::string text;
    for (const auto id : state.schedule->Settings().switches) {
        const bool set = state.environment[static_cast<std::size_t>(id)] != b;
        text += std::string(text.empty() ? "" : ", ") + RuntimeSwitches::Name(id) + (set ? " set" : " unset");
    }
    return text;
}

std::vector<std::string> abAtPresent(AbState& state, std::uint64_t present, double seconds) {
    std::vector<std::string> lines;
    char text[512];
    const auto& config = state.schedule->Settings();
    const auto step = state.schedule->AtPresent(present, seconds);
    if (step.event == AbSchedule::Event::None) return lines;
    if (step.event == AbSchedule::Event::Begin) {
        if (step.phase == 0) {
            char from[96];
            if (config.startSeconds.has_value()) std::snprintf(from, sizeof(from), "%.1f s after the first present", *config.startSeconds);
            else std::snprintf(from, sizeof(from), "present %llu", static_cast<unsigned long long>(config.startPresent));
            std::snprintf(text, sizeof(text), "[ab] in-run A/B: phases of %.1f s from %s, settle %.1f s; A: %s; B: %s\n", config.phaseSeconds, from, config.settleSeconds, SwitchSettings(state, false).c_str(), SwitchSettings(state, true).c_str());
            lines.emplace_back(text);
        }
        for (const auto id : config.switches) RuntimeSwitches::Assign(id, state.environment[static_cast<std::size_t>(id)] != step.b);
        std::snprintf(text, sizeof(text), "[ab] phase %c %llu begins at present %llu (t %.3f s): %s\n", step.b ? 'B' : 'A', static_cast<unsigned long long>(step.phase), static_cast<unsigned long long>(present), seconds, SwitchSettings(state, step.b).c_str());
    } else {
        std::snprintf(text, sizeof(text), "[ab] phase %c %llu settled at present %llu (t %.3f s)\n", step.b ? 'B' : 'A', static_cast<unsigned long long>(step.phase), static_cast<unsigned long long>(present), seconds);
    }
    lines.emplace_back(text);
    return lines;
}

}

bool AbConfigured() {
    static const bool configured = std::getenv("APS5_AB") != nullptr;
    return configured;
}

std::vector<std::string> AbAtPresent(std::uint64_t present, double seconds) {
    if (!AbConfigured()) return {};
    auto& state = Ab();
    std::lock_guard lock(state.mutex);
    return abAtPresent(state, present, seconds);
}

std::vector<std::string> AbAtPresent(std::uint64_t present) {
    if (!AbConfigured()) return {};
    auto& state = Ab();
    std::lock_guard lock(state.mutex);
    return abAtPresent(state, present, std::chrono::duration<double>(std::chrono::steady_clock::now() - state.origin).count());
}

}
