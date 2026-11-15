#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_WRITEVALIDATION_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_WRITEVALIDATION_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace AgcDriver::Graphics {

class Recorder;

}

namespace AgcDriver::Graphics::WriteValidation {

inline constexpr std::uint64_t WindowShift = 16;
inline constexpr std::uint64_t SuperShift = 6;
inline constexpr std::size_t BlockSlots = std::size_t{1} << 17;
inline constexpr std::size_t SuperSlots = std::size_t{1} << 12;
inline constexpr std::uint64_t Unlatched = ~std::uint64_t{0};

enum class Class : std::uint8_t { W, Q, L, C, Count };

enum class Source : std::uint8_t { RecorderNote, InFlightNote, StoragePending, ShadowLive, QueueItem, FinisherJob, RunWrite, PendingJob, Deposit, RecordedLabel, WideLabel, QueuedLabel, BehindCompletion, QueuedStore, QueuedDump, KeyStore, CpuPublish, CleanForget, Count };

enum class Test : std::uint8_t { FollowerInputs, FollowerZones, FollowerStores, FollowerPass, FollowerSelection, FollowerGuards, LeaderInputs, LeaderZones, LeaderInPlace, LeaderGuards, LeaderMoves, LeaderStores, LeaderSelection, RecordMoves, RecordInPlace, TemplateCurrent, ViewSettle, StageSettle, WalkSettle, BuffersSettle, IndexSettle, CapturedMemory, NestedValues, StageInputs, PendingSnapshot, LabelOverlap, Count };

enum class Outcome : std::uint8_t { Answered, Armed, Live, Session, Wide, Evicted, Moved, Snapshot, Watch, Unknown, Stamps, NotClean, Kept, Count };

enum class Answer : std::uint8_t { Today, Proved, Generations };

inline constexpr std::size_t ClassCount = static_cast<std::size_t>(Class::Count);
inline constexpr std::size_t SourceCount = static_cast<std::size_t>(Source::Count);
inline constexpr std::size_t TestCount = static_cast<std::size_t>(Test::Count);
inline constexpr std::size_t OutcomeCount = static_cast<std::size_t>(Outcome::Count);
inline constexpr std::size_t SpanKinds = 3;

namespace Fact {

inline constexpr std::uint16_t Pending = 1u << 0;
inline constexpr std::uint16_t Writes = 1u << 1;
inline constexpr std::uint16_t Storage = 1u << 2;
inline constexpr std::uint16_t Shadows = 1u << 3;
inline constexpr std::uint16_t Labels = 1u << 4;
inline constexpr std::uint16_t Stores = 1u << 5;
inline constexpr std::uint16_t InPlaceExtra = 1u << 6;
inline constexpr std::uint16_t DepthSpans = 1u << 7;
inline constexpr std::uint16_t MultisampleSpans = 1u << 8;
inline constexpr std::uint16_t DepthReads = 1u << 9;
inline constexpr std::uint16_t OfW = Pending | Writes | Storage | Shadows;
inline constexpr std::uint16_t OfL = Labels | Stores | InPlaceExtra;
inline constexpr std::uint16_t OfG = DepthSpans | MultisampleSpans | DepthReads;
inline constexpr std::uint16_t Inputs = Pending | Labels | Stores;
inline constexpr std::uint16_t Moved = Pending | Shadows;
inline constexpr std::uint16_t InPlace = Storage | Shadows | Labels | Stores | InPlaceExtra;
inline constexpr std::uint16_t Guards = DepthReads | MultisampleSpans;

}

struct ClassClocks {
    std::array<std::atomic<std::uint64_t>, BlockSlots> blocks;
    std::array<std::atomic<std::uint64_t>, SuperSlots> supers;
    std::atomic<std::uint64_t> floor;
};

namespace Detail {

extern std::atomic<std::uint64_t> session;
extern std::atomic<std::uint64_t> epoch;
extern std::atomic<ClassClocks*> clocks;
void Raise(Class kind, Source source, std::uint64_t begin, std::uint64_t end);
void RaiseAll(Class kind, Source source);

}

std::uint64_t Latch();
void Relatch();

bool FinisherSide();

inline std::uint64_t Session() {
    const auto value = Detail::session.load(std::memory_order_seq_cst);
    return value != Unlatched ? value : Latch();
}

inline std::uint64_t Epoch() {
    return Detail::epoch.load(std::memory_order_seq_cst);
}

inline std::size_t BlockSlot(std::uint64_t block) {
    return static_cast<std::size_t>((block * 0x9E3779B97F4A7C15ull) >> 47);
}

inline std::size_t SuperSlot(std::uint64_t super) {
    return static_cast<std::size_t>((super * 0x9E3779B97F4A7C15ull) >> 52);
}

inline std::uint64_t Clock(Class kind, std::uint64_t window) {
    auto* all = Detail::clocks.load(std::memory_order_acquire);
    if (all == nullptr) return ~std::uint64_t{0};
    auto& clocks = all[static_cast<std::size_t>(kind)];
    const auto block = clocks.blocks[BlockSlot(window)].load(std::memory_order_seq_cst);
    const auto super = clocks.supers[SuperSlot(window >> SuperShift)].load(std::memory_order_seq_cst);
    const auto floor = clocks.floor.load(std::memory_order_seq_cst);
    return std::max(block, std::max(super, floor));
}

inline void Raise(Class kind, Source source, std::uint64_t begin, std::uint64_t end) {
    if (end > begin && Session() != 0) Detail::Raise(kind, source, begin, end);
}

inline void RaiseAll(Class kind, Source source) {
    if (Session() != 0) Detail::RaiseAll(kind, source);
}

class RaiseScope {
public:
    RaiseScope(Class kind, Source source, std::uint64_t begin, std::uint64_t end) : kind(kind), source(source), begin(begin), end(end) {
        Raise(kind, source, begin, end);
    }
    ~RaiseScope() {
        Raise(kind, source, begin, end);
    }
    RaiseScope(const RaiseScope&) = delete;
    RaiseScope& operator=(const RaiseScope&) = delete;

private:
    Class kind;
    Source source;
    std::uint64_t begin;
    std::uint64_t end;
};

struct Totals {
    std::array<std::array<std::uint64_t, OutcomeCount>, TestCount> outcomes{};
    std::array<std::uint64_t, SourceCount> raises{};
    std::array<std::uint64_t, SpanKinds> spans{};
    std::array<std::uint64_t, TestCount> verified{};
    std::array<std::uint64_t, TestCount> differing{};
};

void Note(Test test, Outcome outcome);
bool VerifyDue(Test test);
void NoteVerify(Test test, bool differs, const char* detail);
std::uint64_t VerifyEvery();
std::uint64_t CountForTest(Test test, Outcome outcome);
Totals Collect();
const char* TestName(Test test);
const char* OutcomeName(Outcome outcome);
const char* SourceName(Source source);
void Report();

Answer CommitterClear(const Recorder& recorder, std::uint64_t begin, std::uint64_t bytes, std::uint16_t facts, Test test);

}

#endif
