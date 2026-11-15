#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/LeanTails.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/LeanFront.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/WaitMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::size_t SpareLimit = 512;
constexpr std::size_t SpareWordsLimit = std::size_t{1} << 20u;
constexpr std::size_t SpareBytesLimit = std::size_t{64} << 20u;

bool listCacheEnabled() {
    static const bool enabled = std::getenv("APS5_LEAN_NO_LIST_CACHE") == nullptr && std::getenv("APS5_SUBMISSION_EPOCH_EACH") == nullptr && !APS5_ENABLE_TIMING_LOG && GuestMemory::WriteWatched();
    return enabled;
}

std::uint32_t controlWord(const std::uint32_t* tail) {
    return std::atomic_ref<std::uint32_t>(*const_cast<std::uint32_t*>(tail - 1)).load(std::memory_order_relaxed);
}

bool sameRewindPayload(std::uint32_t scanned, const std::vector<std::uint32_t>& commands) {
    const auto now = commands.back();
    if (scanned == now) return true;
    if ((scanned ^ now) != Pm4::RewindValid || commands.size() < 2) return false;
    const auto header = commands[commands.size() - 2];
    return (header >> 30u) == 3u && ((header >> 8u) & 0xffu) == 0x59u && Pm4::PacketWords(header) == 2u;
}

std::size_t commandBytes(const std::vector<std::uint32_t>& commands) {
    return commands.capacity() * sizeof(std::uint32_t);
}

std::size_t listBytes(const RegisterLists& lists) {
    return lists.words.capacity() * sizeof(std::uint32_t) + lists.refs.capacity() * sizeof(RegisterListRef);
}

std::string hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

std::string conditionalText(const std::map<std::size_t, std::size_t>& ends) {
    std::string text;
    for (const auto& [from, to] : ends) {
        if (text.size() > 96) {
            text += " ...";
            break;
        }
        text += (text.empty() ? "" : " ") + std::to_string(from) + "->" + std::to_string(to);
    }
    return text.empty() ? std::string("none") : text;
}

void compareKeptTail(const Submission& kept, const Submission& fresh, const Submission& stalled) {
    const auto where = "serial " + std::to_string(stalled.serial) + " tail " + hex(reinterpret_cast<std::uintptr_t>(stalled.rewindTail));
    const auto& left = kept.commands;
    const auto& right = fresh.commands;
    const auto masked = [&](std::size_t index, std::uint32_t value) { return fresh.rewindTail != nullptr && index + 1 == right.size() ? value & ~Pm4::RewindValid : value; };
    std::size_t differs = std::min(left.size(), right.size());
    for (std::size_t index = 0; index < std::min(left.size(), right.size()); ++index) {
        if (masked(index, left[index]) != masked(index, right[index])) {
            differs = index;
            break;
        }
    }
    if (left.size() != right.size() || differs != left.size()) {
        auto detail = where + ": the copy kept at the header has " + std::to_string(left.size()) + " dwords, today's copy " + std::to_string(right.size());
        if (differs < left.size() && differs < right.size()) detail += "; first difference at dword " + std::to_string(differs) + ": kept " + hex(left[differs]) + ", today " + hex(right[differs]);
        LeanFront::NoteDifference(LeanFront::Field::SegmentWords, detail);
    }
    if (kept.conditionalEnds != fresh.conditionalEnds) LeanFront::NoteDifference(LeanFront::Field::ConditionalEnds, where + ": kept " + conditionalText(kept.conditionalEnds) + ", today " + conditionalText(fresh.conditionalEnds));
    if (kept.rewindTail != fresh.rewindTail || kept.rewindWords != fresh.rewindWords) {
        LeanFront::NoteDifference(LeanFront::Field::RewindTail, where + ": kept next tail " + hex(reinterpret_cast<std::uintptr_t>(kept.rewindTail)) + " with " + std::to_string(kept.rewindWords) + " dwords, today " + hex(reinterpret_cast<std::uintptr_t>(fresh.rewindTail)) + " with " + std::to_string(fresh.rewindWords));
    }
}

std::uint64_t nanosecondsBetween(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

std::string packetText(const LeanPacket& entry) {
    return "cursor " + std::to_string(entry.cursor) + " words " + std::to_string(entry.words) + " header " + hex(entry.header) + " opcode " + hex(entry.opcode) + " skip " + std::to_string(entry.skip) + " flags " + hex(entry.flags) + " pipeline " + std::to_string(static_cast<unsigned>(entry.pipeline));
}

std::string tableDifference(const PacketTable& ahead, const PacketTable& today) {
    if (ahead.packets.size() != today.packets.size()) return "the packet table lists " + std::to_string(ahead.packets.size()) + " packets, today's scan " + std::to_string(today.packets.size());
    for (std::size_t index = 0; index < ahead.packets.size(); ++index) {
        const auto& left = ahead.packets[index];
        const auto& right = today.packets[index];
        if (left.cursor == right.cursor && left.words == right.words && left.header == right.header && left.opcode == right.opcode && left.skip == right.skip && left.flags == right.flags && left.pipeline == right.pipeline) continue;
        return "packet table entry " + std::to_string(index) + ": ahead " + packetText(left) + ", today " + packetText(right);
    }
    if (ahead.outputs != today.outputs) return "the output positions differ: " + std::to_string(ahead.outputs.size()) + " ahead, " + std::to_string(today.outputs.size()) + " today";
    if (ahead.lists != today.lists) return "the register list positions differ: " + std::to_string(ahead.lists.size()) + " ahead, " + std::to_string(today.lists.size()) + " today";
    if (ahead.flipHeaders != today.flipHeaders) return "flip headers: " + std::to_string(ahead.flipHeaders) + " ahead, " + std::to_string(today.flipHeaders) + " today";
    if (ahead.built != today.built) return std::string("the built mark: ") + (ahead.built ? "set" : "unset") + " ahead, " + (today.built ? "set" : "unset") + " today";
    return {};
}

std::string commandsDifference(const Submission& tail, const Submission& fresh) {
    const auto& left = tail.commands;
    const auto& right = fresh.commands;
    const auto masked = [&](std::size_t index, std::uint32_t value) { return fresh.rewindTail != nullptr && index + 1 == right.size() ? value & ~Pm4::RewindValid : value; };
    if (left.size() != right.size()) return "the commands have " + std::to_string(left.size()) + " dwords, today's copy " + std::to_string(right.size());
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (masked(index, left[index]) != masked(index, right[index])) return "first command difference at dword " + std::to_string(index) + ": ahead " + hex(left[index]) + ", today " + hex(right[index]);
    }
    if (tail.conditionalEnds != fresh.conditionalEnds) return "conditional ends: ahead " + conditionalText(tail.conditionalEnds) + ", today " + conditionalText(fresh.conditionalEnds);
    if (tail.rewindTail != fresh.rewindTail || tail.rewindWords != fresh.rewindWords) return "next tail: ahead " + hex(reinterpret_cast<std::uintptr_t>(tail.rewindTail)) + " with " + std::to_string(tail.rewindWords) + " dwords, today " + hex(reinterpret_cast<std::uintptr_t>(fresh.rewindTail)) + " with " + std::to_string(fresh.rewindWords);
    return {};
}

}

namespace TailsAhead {

namespace {

constexpr std::size_t Counts = static_cast<std::size_t>(Count::Kinds);
constexpr std::int64_t WindowMs = 10000;
constexpr std::uint32_t PrintedPerWindow = 16;

struct Census {
    std::array<std::atomic<std::uint64_t>, Counts> values{};
    std::array<std::uint64_t, Counts> reported{};
    std::int64_t reportedAt = 0;
    std::atomic<std::uint32_t> printed{0};
    std::mutex mutex;
};

Census& census() {
    static auto* state = new Census;
    return *state;
}

double millis(std::uint64_t nanoseconds) {
    return static_cast<double>(nanoseconds) / 1e6;
}

double micros(std::uint64_t nanoseconds, std::uint64_t count) {
    return count == 0 ? 0.0 : static_cast<double>(nanoseconds) / 1e3 / static_cast<double>(count);
}

}

bool Stats() {
    static const bool on = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    return on;
}

std::uint64_t VerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_TAILS_AHEAD");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

void Add(Count count, std::uint64_t value) {
    if (!Stats() && VerifyEvery() == 0) return;
    census().values[static_cast<std::size_t>(count)].fetch_add(value, std::memory_order_relaxed);
}

void NoteDifference(const std::string& detail) {
    auto& state = census();
    state.values[static_cast<std::size_t>(Count::VerifyDiffering)].fetch_add(1, std::memory_order_relaxed);
    if (state.printed.fetch_add(1, std::memory_order_relaxed) < PrintedPerWindow) std::fprintf(stderr, "[tails-ahead-verify] %s\n", detail.c_str());
}

void Report() {
    const bool stats = Stats();
    const auto every = VerifyEvery();
    if (!stats && every == 0) return;
    auto& state = census();
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    const std::unique_lock lock(state.mutex, std::try_to_lock);
    if (!lock.owns_lock()) return;
    const auto last = state.reportedAt;
    if (last == 0) {
        state.reportedAt = now;
        return;
    }
    if (now - last < WindowMs) return;
    state.reportedAt = now;
    const double seconds = static_cast<double>(now - last) / 1000.0;
    std::array<std::uint64_t, Counts> window{};
    for (std::size_t index = 0; index < Counts; ++index) {
        const auto value = state.values[index].load(std::memory_order_relaxed);
        window[index] = value - state.reported[index];
        state.reported[index] = value;
    }
    state.printed.store(0, std::memory_order_relaxed);
    const auto at = [&](Count count) { return window[static_cast<std::size_t>(count)]; };
    const auto shown = [&](Count count) { return static_cast<unsigned long long>(at(count)); };
    if (stats) {
        AgcDriver::ProfilePrint_nid_no_patch("[tails-ahead] %.1f s: handed to the helper %llu, no free job %llu; prepared ahead %llu in %.2f ms of the helper: tables %llu (refused by the scan %llu), lists %llu (not built: list shape %llu, unreadable %llu, not watched %llu, forget in progress %llu); used: tables %llu, lists %llu; today's path instead: not ready in time %llu (not started %llu, still running %llu), sources changed %llu, forget stamp moved %llu, a driver store over the sources %llu, front mode changed %llu, control word moved %llu, verify %llu, hooks would run %llu, lists not built ahead %llu, other %llu, dropped unused %llu; moved off the worker %.2f ms (%.2f us per used tail); worker tail preparation: ahead %llu in %.2f ms (%.2f us each), partly ahead %llu in %.2f ms (%.2f us each), today's %llu in %.2f ms (%.2f us each)\n",
            seconds, shown(Count::Handed), shown(Count::NoJob), shown(Count::Prepared), millis(at(Count::HelperNs)), shown(Count::TablesBuilt), shown(Count::TableFailed), shown(Count::ListsBuilt), shown(Count::ListsShape), shown(Count::ListsUnreadable), shown(Count::ListsUnwatched), shown(Count::ListsForgetting),
            shown(Count::TablesUsed), shown(Count::ListsUsed), static_cast<unsigned long long>(at(Count::NotStarted) + at(Count::StillRunning)), shown(Count::NotStarted), shown(Count::StillRunning), shown(Count::SourcesChanged), shown(Count::ForgetMoved), shown(Count::DriverStore), shown(Count::FrontChanged), shown(Count::ControlWordMoved), shown(Count::Verify), shown(Count::HooksWouldRun), shown(Count::ListsNotBuilt), shown(Count::Other), shown(Count::Dropped),
            millis(at(Count::MovedNs)), micros(at(Count::MovedNs), at(Count::TablesUsed)), shown(Count::AheadTails), millis(at(Count::AheadNs)), micros(at(Count::AheadNs), at(Count::AheadTails)), shown(Count::PartialTails), millis(at(Count::PartialNs)), micros(at(Count::PartialNs), at(Count::PartialTails)), shown(Count::TodayTails), millis(at(Count::TodayNs)), micros(at(Count::TodayNs), at(Count::TodayTails)));
    }
    if (every != 0) AgcDriver::ProfilePrint_nid_no_patch("[tails-ahead-verify] %.1f s: every %llu: tails checked %llu, differences %llu\n", seconds, static_cast<unsigned long long>(every), shown(Count::VerifyChecked), shown(Count::VerifyDiffering));
}

}

AheadTails::~AheadTails() {
    Stop();
}

AheadTail* AheadTails::Acquire() {
    if (jobs == nullptr) {
        jobs = std::make_unique<AheadTail[]>(Jobs);
        ring = std::make_unique<AheadTail*[]>(RingSlots);
    }
    for (std::size_t tries = 0; tries < 8; ++tries) {
        auto& job = jobs[cursor];
        cursor = (cursor + 1) % Jobs;
        if (job.state.load(std::memory_order_acquire) == AheadTail::Free) return &job;
    }
    return nullptr;
}

bool AheadTails::Publish(AheadTail& job, Work prepare) {
    if (failed || stopping.load(std::memory_order_acquire)) return false;
    const auto at = head.load(std::memory_order_relaxed);
    if (at - tail.load(std::memory_order_acquire) >= RingSlots) return false;
    if (!started) {
        started = true;
        work = prepare;
        try {
            thread = std::thread([this] { run(); });
        } catch (const std::system_error&) {
            failed = true;
            return false;
        }
    }
    ring[at % RingSlots] = &job;
    job.state.store(AheadTail::Queued, std::memory_order_release);
    head.store(at + 1, std::memory_order_release);
    return true;
}

void AheadTails::Wake() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!sleeping.load(std::memory_order_seq_cst)) return;
    const std::unique_lock lock(mutex, std::try_to_lock);
    wake.notify_one();
}

void AheadTails::Stop() {
    if (!thread.joinable()) return;
    {
        std::lock_guard lock(mutex);
        stopping.store(true, std::memory_order_release);
    }
    wake.notify_all();
    if (thread.get_id() == std::this_thread::get_id()) thread.detach();
    else thread.join();
}

void AheadTails::run() {
    PinWorkerThread("tails ahead");
    for (;;) {
        if (stopping.load(std::memory_order_acquire)) return;
        const auto at = tail.load(std::memory_order_relaxed);
        if (at == head.load(std::memory_order_acquire)) {
            sleeping.store(true, std::memory_order_seq_cst);
            if (at == head.load(std::memory_order_seq_cst) && !stopping.load(std::memory_order_acquire)) {
                std::unique_lock lock(mutex);
                wake.wait_for(lock, std::chrono::milliseconds(1), [&] { return at != head.load(std::memory_order_acquire) || stopping.load(std::memory_order_acquire); });
            }
            sleeping.store(false, std::memory_order_relaxed);
            continue;
        }
        auto* job = ring[at % RingSlots];
        tail.store(at + 1, std::memory_order_release);
        auto expected = AheadTail::Queued;
        if (!job->state.compare_exchange_strong(expected, AheadTail::Running, std::memory_order_acq_rel)) continue;
        try {
            work(*job);
        } catch (...) {
            job->tableBuilt = false;
            job->lists = AheadTail::Lists::NotWanted;
        }
        expected = AheadTail::Running;
        if (!job->state.compare_exchange_strong(expected, AheadTail::Done, std::memory_order_acq_rel)) job->state.store(AheadTail::Free, std::memory_order_release);
    }
}

void LeanTails::Attach(KeptTail& slot, AheadTail& job) {
    slot.job = &job;
    ++attached;
}

AheadTail* LeanTails::Reclaim(KeptTail& slot) {
    auto* job = slot.job;
    if (job == nullptr) return nullptr;
    slot.job = nullptr;
    --attached;
    const auto giveBack = [&] {
        std::swap(slot.copy.commands, job->copy.commands);
        std::swap(slot.copy.conditionalEnds, job->copy.conditionalEnds);
    };
    auto state = job->state.load(std::memory_order_acquire);
    for (;;) {
        if (state == AheadTail::Queued) {
            if (!job->state.compare_exchange_weak(state, AheadTail::Free, std::memory_order_acq_rel)) continue;
            giveBack();
            TailsAhead::Add(TailsAhead::Count::NotStarted);
            return nullptr;
        }
        if (state == AheadTail::Running) {
            if (!job->state.compare_exchange_weak(state, AheadTail::Abandoned, std::memory_order_acq_rel)) continue;
            const auto& words = job->copy.commands;
            const bool refreshed = job->copy.rewindTail != nullptr && !words.empty();
            slot.copy.commands.assign(words.begin(), refreshed ? words.end() - 1 : words.end());
            if (refreshed) slot.copy.commands.push_back(controlWord(job->copy.rewindTail));
            slot.copy.conditionalEnds = job->copy.conditionalEnds;
            TailsAhead::Add(TailsAhead::Count::StillRunning);
            return nullptr;
        }
        if (state == AheadTail::Done) {
            giveBack();
            return job;
        }
        return nullptr;
    }
}

void LeanTails::Release(AheadTail& job) {
    job.state.store(AheadTail::Free, std::memory_order_release);
}

void LeanTails::detach(KeptTail& slot) {
    auto* job = slot.job;
    if (job == nullptr) return;
    slot.job = nullptr;
    --attached;
    auto state = job->state.load(std::memory_order_acquire);
    for (;;) {
        if (state == AheadTail::Queued) {
            if (job->state.compare_exchange_weak(state, AheadTail::Free, std::memory_order_acq_rel)) break;
            continue;
        }
        if (state == AheadTail::Running) {
            if (job->state.compare_exchange_weak(state, AheadTail::Abandoned, std::memory_order_acq_rel)) break;
            continue;
        }
        if (state == AheadTail::Done) job->state.store(AheadTail::Free, std::memory_order_release);
        break;
    }
    TailsAhead::Add(TailsAhead::Count::Dropped);
}

KeptTail& LeanTails::Keep() {
    if (keptFront == keptEnd) keptFront = keptEnd = 0;
    if (keptEnd == kept.size()) kept.emplace_back();
    auto& slot = kept[keptEnd++];
    if (slot.copy.commands.capacity() == 0 || slot.copy.commands.capacity() > SpareWordsLimit) slot.copy.commands = TakeCommands();
    slot.copy.rewindTail = nullptr;
    slot.copy.rewindWords = 0;
    slot.sources.clear();
    return slot;
}

void LeanTails::Unkeep() {
    if (keptEnd != keptFront) --keptEnd;
}

KeptTail* LeanTails::Front(const std::uint32_t* control) {
    if (keptFront == keptEnd || kept[keptFront].control != control) return nullptr;
    return &kept[keptFront];
}

void LeanTails::Pop() {
    if (keptFront == keptEnd) return;
    if (attached != 0) detach(kept[keptFront]);
    if (++keptFront == keptEnd) keptFront = keptEnd = 0;
}

void LeanTails::ClearKept() {
    if (attached != 0) {
        for (auto& slot : kept) detach(slot);
    }
    keptFront = keptEnd = 0;
}

void LeanTails::TrimKept() {
    for (std::size_t index = 0; index < kept.size(); ++index) {
        auto& commands = kept[index].copy.commands;
        if ((index >= keptFront && index < keptEnd) || commands.capacity() == 0) continue;
        GiveCommands(std::move(commands));
        std::vector<std::uint32_t>().swap(commands);
    }
}

std::vector<std::uint32_t> LeanTails::TakeCommands() {
    if (spareCommands.empty()) return {};
    auto commands = std::move(spareCommands.back());
    spareCommands.pop_back();
    spareBytes -= commandBytes(commands);
    return commands;
}

void LeanTails::GiveCommands(std::vector<std::uint32_t>&& commands) {
    const auto bytes = commandBytes(commands);
    if (commands.capacity() == 0 || commands.capacity() > SpareWordsLimit || spareCommands.size() >= SpareLimit || spareBytes + bytes > SpareBytesLimit) return;
    commands.clear();
    spareBytes += bytes;
    spareCommands.push_back(std::move(commands));
}

RegisterLists LeanTails::TakeLists() {
    if (spareLists.empty()) return {};
    auto lists = std::move(spareLists.back());
    spareLists.pop_back();
    spareBytes -= listBytes(lists);
    return lists;
}

void LeanTails::GiveLists(RegisterLists&& lists) {
    const auto bytes = listBytes(lists);
    if (lists.words.capacity() > SpareWordsLimit || spareLists.size() >= SpareLimit || spareBytes + bytes > SpareBytesLimit) return;
    lists.Clear();
    spareBytes += bytes;
    spareLists.push_back(std::move(lists));
}

void Driver::keepReleasedTails(const Submission& submission) {
    auto& released = releasedTails();
    const auto tracked = GuestMemory::TrackerGeneration();
    const auto forgets = GuestMemory::ForgetSerial();
    const bool ahead = frontLatch.switches.Set(Graphics::RuntimeSwitch::TailsAhead);
    const bool byTable = ahead && frontLatch.switches.Set(Graphics::RuntimeSwitch::PacketClassTable);
    const bool wantLists = ahead && listsAheadWanted();
    bool handed = false;
    const std::uint32_t* tail = submission.rewindTail;
    std::size_t words = submission.rewindWords;
    for (std::size_t chunk = 0; chunk < 256 && tail != nullptr; ++chunk) {
        if ((std::atomic_ref<std::uint32_t>(*const_cast<std::uint32_t*>(tail - 1)).load(std::memory_order_acquire) & 0x80000000u) == 0) break;
        released.push_back(tail - 1);
        auto& kept = leanTails.Keep();
        kept.control = tail - 1;
        kept.words = words;
        kept.tracked = tracked;
        kept.forgets = forgets;
        kept.stamp = frontLatch.generation;
        try {
            copyCommands(kept.copy, tail, words, &kept.sources);
        } catch (const std::exception&) {
            leanTails.Unkeep();
            break;
        }
        tail = kept.copy.rewindTail;
        words = kept.copy.rewindWords;
        if (ahead && handTailAhead(kept, byTable, wantLists)) handed = true;
    }
    if (handed) leanTails.ahead.Wake();
    leanTails.TrimKept();
}

bool Driver::handTailAhead(KeptTail& kept, bool byTable, bool wantLists) {
    using TailsAhead::Count;
    auto* job = leanTails.ahead.Acquire();
    if (job == nullptr) {
        TailsAhead::Add(Count::NoJob);
        return false;
    }
    const auto swapCopy = [&] {
        std::swap(job->copy.commands, kept.copy.commands);
        std::swap(job->copy.conditionalEnds, kept.copy.conditionalEnds);
    };
    swapCopy();
    kept.copy.commands.clear();
    kept.copy.conditionalEnds.clear();
    job->copy.queue = 0;
    job->copy.rewindTail = kept.copy.rewindTail;
    job->copy.rewindWords = kept.copy.rewindWords;
    job->stamp = kept.stamp;
    job->byTable = byTable;
    job->wantLists = wantLists;
    job->tableBuilt = false;
    job->refreshed = false;
    job->lastWord = 0;
    job->lists = AheadTail::Lists::NotWanted;
    job->tableNs = 0;
    job->listsNs = 0;
    if (wantLists && job->resolved.words.capacity() == 0) job->resolved = leanTails.TakeLists();
    if (!leanTails.ahead.Publish(*job, &Driver::prepareTailAhead)) {
        swapCopy();
        TailsAhead::Add(Count::NoJob);
        return false;
    }
    leanTails.Attach(kept, *job);
    TailsAhead::Add(Count::Handed);
    return true;
}

void Driver::prepareTailAhead(AheadTail& job) {
    using TailsAhead::Count;
    const auto started = std::chrono::steady_clock::now();
    auto& copy = job.copy;
    job.refreshed = copy.rewindTail != nullptr && !copy.commands.empty();
    struct LastWord {
        std::vector<std::uint32_t>& commands;
        bool refreshed;
        std::uint32_t original;
        ~LastWord() {
            if (refreshed) commands.back() = original;
        }
    } restore{copy.commands, job.refreshed, job.refreshed ? copy.commands.back() : 0u};
    if (job.refreshed) copy.commands.back() = job.lastWord = controlWord(copy.rewindTail);
    try {
        scanSegment(copy, job.table, true, nullptr, job.byTable, true);
        job.tableBuilt = true;
        TailsAhead::Add(Count::TablesBuilt);
    } catch (const std::exception&) {
        job.table.Clear();
        job.tableBuilt = false;
        TailsAhead::Add(Count::TableFailed);
    }
    const auto scanned = std::chrono::steady_clock::now();
    if (job.wantLists && job.tableBuilt) {
        job.lists = ResolveListsAhead(job);
        switch (job.lists) {
            case AheadTail::Lists::Built: TailsAhead::Add(Count::ListsBuilt); break;
            case AheadTail::Lists::Shape: TailsAhead::Add(Count::ListsShape); break;
            case AheadTail::Lists::Unreadable: TailsAhead::Add(Count::ListsUnreadable); break;
            case AheadTail::Lists::Unwatched: TailsAhead::Add(Count::ListsUnwatched); break;
            case AheadTail::Lists::Forgetting: TailsAhead::Add(Count::ListsForgetting); break;
            default: break;
        }
    } else {
        job.lists = job.wantLists ? AheadTail::Lists::NoTable : AheadTail::Lists::NotWanted;
    }
    const auto finished = std::chrono::steady_clock::now();
    job.tableNs = nanosecondsBetween(started, scanned);
    job.listsNs = nanosecondsBetween(scanned, finished);
    TailsAhead::Add(Count::Prepared);
    TailsAhead::Add(Count::HelperNs, nanosecondsBetween(started, finished));
}

bool Driver::verifyAheadTable(const Submission& stalled, const Submission& tail, PacketTable& table, bool byTable) {
    TailsAhead::Add(TailsAhead::Count::VerifyChecked);
    const auto where = "serial " + std::to_string(stalled.serial) + " tail " + hex(reinterpret_cast<std::uintptr_t>(stalled.rewindTail));
    Submission fresh{};
    fresh.queue = tail.queue;
    try {
        copyCommands(fresh, stalled.rewindTail, stalled.rewindWords);
        if (const auto difference = commandsDifference(tail, fresh); !difference.empty()) TailsAhead::NoteDifference(where + ": commands (the kept copy, which today's preparation takes as well): " + difference);
    } catch (const std::exception& error) {
        TailsAhead::NoteDifference(where + ": today's copy of the commands failed: " + error.what());
    }
    PacketTable today;
    try {
        scanSegment(tail, today, true, stalled.rewindTail, byTable);
    } catch (const std::exception& error) {
        TailsAhead::NoteDifference(where + ": the helper's scan accepted the tail, today's refused it: " + error.what() + "; the tail takes today's result");
        throw;
    }
    const auto difference = tableDifference(table, today);
    if (difference.empty()) return true;
    TailsAhead::NoteDifference(where + ": " + difference + "; the tail takes today's table and lists");
    std::swap(table, today);
    return false;
}

TailsAhead::Count Driver::keptRefusal(const KeptTail& kept, const Submission& stalled) const {
    using TailsAhead::Count;
    if (kept.stamp != frontLatch.generation) return Count::FrontChanged;
    if (kept.words != stalled.rewindWords) return Count::SourcesChanged;
    for (const auto& [address, bytes] : kept.sources) {
        if (GuestMemory::ForgetStamp(address, static_cast<std::size_t>(bytes)) > kept.forgets) return Count::ForgetMoved;
    }
    return Count::DriverStore;
}

bool Driver::keptTailCurrent(const KeptTail& kept, const Submission& stalled) const {
    if (kept.stamp != frontLatch.generation || kept.words != stalled.rewindWords) return false;
    for (const auto& [address, bytes] : kept.sources) {
        if (GuestMemory::ForgetStamp(address, static_cast<std::size_t>(bytes)) > kept.forgets) return false;
    }
    return !GuestMemory::DriverStoredOver(kept.sources, kept.tracked);
}

void Driver::recycleLeanTail(Submission& tail) {
    if (tail.queue != 0) return;
    leanTails.GiveCommands(std::move(tail.commands));
    if (tail.leanLists) leanTails.GiveLists(std::move(tail.lists));
}

void Driver::prepareLeanTail(const Submission& stalled, Submission& tail, bool popped, bool verify, PacketTable* table) {
    using LeanFront::Counter;
    using TailsAhead::Count;
    const auto control = stalled.rewindTail - 1;
    if (frontLatch.Renew(leanTails.cache.stamp)) leanTails.cache.Clear();
    const bool aheadOn = frontLatch.switches.Set(Graphics::RuntimeSwitch::TailsAhead);
    const bool timed = TailsAhead::Stats();
    const auto startedAt = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    KeptTail* kept = nullptr;
    if (leanTails.Kept()) {
        kept = leanTails.Front(control);
        if (verify && (popped ? kept == nullptr : kept != nullptr)) {
            LeanFront::NoteDifference(LeanFront::Field::ReleasedTails, "serial " + std::to_string(stalled.serial) + " tail " + hex(reinterpret_cast<std::uintptr_t>(stalled.rewindTail)) + ": the released tails " + (popped ? "list it next" : "do not list it next") + ", the kept copies " + (kept != nullptr ? "do" : "do not"));
        }
        if (!popped || kept == nullptr) {
            leanTails.ClearKept();
            kept = nullptr;
        }
    }
    AheadTail* ahead = nullptr;
    if (kept != nullptr && kept->job != nullptr) ahead = leanTails.Reclaim(*kept);
    struct HeldAhead {
        LeanTails& tails;
        AheadTail* job;
        ~HeldAhead() {
            if (job != nullptr) tails.Release(*job);
        }
    } held{leanTails, ahead};
    const bool current = kept != nullptr && keptTailCurrent(*kept, stalled);
    if (kept == nullptr) LeanFront::AddOnWorker(Counter::PreparedCopiesMissed);
    else if (!current) LeanFront::AddOnWorker(Counter::PreparedCopiesRefused);
    std::optional<Count> refused;
    if (ahead != nullptr) {
        if (!aheadOn) refused = Count::FrontChanged;
        else if (verify || frontLatch.verifyEvery != 0) refused = Count::Verify;
        else if (!current) refused = keptRefusal(*kept, stalled);
        else if (table == nullptr) refused = Count::Other;
    }
    if (current && !verify) {
        std::swap(tail.commands, kept->copy.commands);
        tail.conditionalEnds = std::move(kept->copy.conditionalEnds);
        kept->copy.conditionalEnds.clear();
        tail.rewindTail = kept->copy.rewindTail;
        tail.rewindWords = kept->copy.rewindWords;
        if (tail.rewindTail != nullptr && !tail.commands.empty()) tail.commands.back() = controlWord(tail.rewindTail);
        LeanFront::AddOnWorker(Counter::PreparedCopiesUsed);
    } else {
        tail.commands = leanTails.TakeCommands();
        copyCommands(tail, stalled.rewindTail, stalled.rewindWords);
        if (verify) {
            LeanFront::NoteChecked(LeanFront::Item::Segment);
            if (current) compareKeptTail(kept->copy, tail, stalled);
        }
    }
    if (kept != nullptr) leanTails.Pop();
    if (APS5_ENABLE_TIMING_LOG) tail.copiedAt = std::chrono::steady_clock::now();
    const bool byTable = table != nullptr && !verify && LeanFront::SwitchSet(frontLatch, Graphics::RuntimeSwitch::PacketClassTable);
    bool matched = false;
    if (ahead != nullptr && !refused) {
        const bool refresh = tail.rewindTail != nullptr && !tail.commands.empty();
        if (!ahead->tableBuilt) refused = Count::Other;
        else if (ahead->byTable != byTable || ahead->stamp != frontLatch.generation) refused = Count::FrontChanged;
        else if (ahead->refreshed != refresh || (refresh && !sameRewindPayload(ahead->lastWord, tail.commands))) refused = Count::ControlWordMoved;
        else matched = true;
    }
    const auto verifyEvery = TailsAhead::VerifyEvery();
    const bool checked = matched && verifyEvery != 0 && ++leanTails.aheadVerifyOrdinal % verifyEvery == 0;
    bool tableTaken = false;
    if (table != nullptr && !verify) {
        if (matched) {
            std::swap(*table, ahead->table);
            tableTaken = true;
            if (checked && !verifyAheadTable(stalled, tail, *table, byTable)) {
                tableTaken = false;
                matched = false;
            }
        } else {
            scanSegment(tail, *table, true, stalled.rewindTail, byTable);
        }
        LeanFront::AddOnWorker(Counter::PacketTables);
    } else {
        validate(tail, stalled.rewindTail);
    }
    const bool cacheable = stalled.queue == 0 && listCacheEnabled();
    if (cacheable && !popped) {
        noteReleasedTails(tail, FrontMode::Lean);
        bumpEpoch(&EpochBumps::submissions);
        tail.enqueuedAt = lastEpochBump();
        LeanFront::AddOnWorker(Counter::LateTailHoists);
    }
    const bool cache = cacheable && GuestMemory::ThreadCollectEpoch() != 0;
    bool listsTaken = false;
    if (verify) {
        const auto ownWrites = GuestMemory::OwnWriteSerial();
        readRegisterLists(tail);
        verifyLeanLists(tail, cache, ownWrites);
    } else {
        if (matched) {
            if (ahead->lists == AheadTail::Lists::Built) listsTaken = takeAheadLists(tail, cache, *ahead);
            else TailsAhead::Add(ahead->wantLists ? Count::ListsNotBuilt : Count::HooksWouldRun);
        }
        if (listsTaken && checked) {
            std::string detail;
            if (!CheckAheadLists(tail, byTable ? table : nullptr, detail)) {
                TailsAhead::NoteDifference("serial " + std::to_string(stalled.serial) + " tail " + hex(reinterpret_cast<std::uintptr_t>(stalled.rewindTail)) + ": register lists: " + detail + "; the tail takes today's lists");
                listsTaken = false;
            }
        }
        if (!listsTaken) resolveLeanLists(tail, cache, byTable ? table : nullptr);
    }
    if (ahead != nullptr) {
        if (refused) TailsAhead::Add(*refused);
        if (tableTaken) {
            TailsAhead::Add(Count::TablesUsed);
            TailsAhead::Add(Count::MovedNs, ahead->tableNs);
        }
        if (listsTaken) {
            TailsAhead::Add(Count::ListsUsed);
            TailsAhead::Add(Count::MovedNs, ahead->listsNs);
        }
    }
    if (timed) {
        const auto spent = nanosecondsBetween(startedAt, std::chrono::steady_clock::now());
        if (tableTaken && listsTaken) {
            TailsAhead::Add(Count::AheadTails);
            TailsAhead::Add(Count::AheadNs, spent);
        } else if (tableTaken || listsTaken) {
            TailsAhead::Add(Count::PartialTails);
            TailsAhead::Add(Count::PartialNs, spent);
        } else {
            TailsAhead::Add(Count::TodayTails);
            TailsAhead::Add(Count::TodayNs, spent);
        }
    }
    if (timed || (aheadOn && verifyEvery != 0)) TailsAhead::Report();
}

}
