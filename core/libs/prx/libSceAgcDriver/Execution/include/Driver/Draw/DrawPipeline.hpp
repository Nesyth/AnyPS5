#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWPIPELINE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWPIPELINE_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DeepIndex.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

class PublishedRanges {
public:
    using Range = std::pair<std::uint64_t, std::uint64_t>;
    static constexpr std::uint32_t Capacity = 32;
    static constexpr std::uint32_t UnknownCapacity = 8;
    enum class Answer : std::uint8_t { Clear, Hit, Unsure };
    void Publish(std::span<const std::pair<Range, std::uint32_t>> entries, std::uint32_t unknownHeld);
    void PublishUnknownJobs(std::span<const std::uint64_t> jobs);
    Answer Overlaps(std::span<const Range> ranges, Range* pendingHit = nullptr, bool* unknownHeld = nullptr, bool ignoreUnknown = false) const;
    bool UnknownAhead(std::uint64_t front, std::uint64_t asker, std::uint64_t& blocker) const;
    bool UnknownHeld() const;
    bool Copy(std::array<Range, Capacity>& out, std::uint32_t& size) const;

private:
    std::atomic<std::uint32_t> seq{0};
    std::atomic<std::uint32_t> count{0};
    std::atomic<std::uint32_t> unknown{0};
    std::atomic<std::uint64_t> lowest{0};
    std::atomic<std::uint64_t> highest{0};
    std::array<std::atomic<std::uint64_t>, Capacity * 2> bounds{};
    std::atomic<std::uint32_t> unknownJobCount{0};
    std::array<std::atomic<std::uint64_t>, UnknownCapacity> unknownJobs{};
};

class PublishedKeys {
public:
    static constexpr std::uint32_t Capacity = 64;
    static constexpr std::uint32_t DeepDepth = 64;
    static constexpr std::uint64_t AnyKey = ~std::uint64_t{0};
    enum class Answer : std::uint8_t { Clear, Hit, Unsure };
    void Push(std::uint64_t seqNumber, std::uint64_t key, std::uint64_t shape, bool inserts);
    void Retire(std::uint64_t frontSeq);
    Answer Find(std::uint64_t key, std::uint64_t shape, bool insertsOnly) const;
    Answer FindBefore(std::uint64_t key, std::uint64_t shape, bool insertsOnly, std::uint64_t before) const;

private:
    std::atomic<std::uint32_t> seq{0};
    std::atomic<std::uint64_t> head{1};
    std::atomic<std::uint64_t> tail{1};
    std::array<std::atomic<std::uint64_t>, Capacity * 3> slots{};
};

struct LabelRun {
    struct Label {
        std::uint64_t address = 0;
        std::uint32_t offset = 0;
        std::uint32_t size = 0;
        std::uint32_t opcode = 0;
        std::uint32_t cursor = 0;
        bool stores = false;
        std::optional<std::uint32_t> interrupt;
    };
    enum class Lookup : std::uint8_t { NotHeld, Value, Unknown };
    std::vector<Label> labels;
    std::vector<std::byte> bytes;

    std::span<const std::byte> BytesOf(const Label& label) const { return std::span<const std::byte>(bytes).subspan(label.offset, label.size); }
    Lookup Find(std::uint64_t address, std::size_t size, std::uint64_t& value) const;
};

enum class DeepSite : std::uint32_t { PacketVerdict, Predication, IndirectArguments, FillInputs, FillByTurn, DispatchReads, LateDispatchReads, Decode, HeapStages, Variant, WriteEvidence, ShaderCode, SameKey, EarlierKey, ItemsByTurn, ItemsWithReadsByTurn, PacketFinisher, Count };
enum class DeepKept : std::uint32_t { Flip, RenderingWait, Wait, WorkerLabel, PacketVerdict, SegmentEnd, Submission, Settle, DeferredLabels, WorkerDraw, WorkerDispatch, MetadataPass, WorkerNote, Relocation, IndirectCpu, DispatchNotPipelined, Fallback, Shutdown, Other, NoteNotCandidate, NoteMultisampleClear, NoteMemoryValueFill, NoteHtileDepthClear, NoteHtileMaskClear, NoteHtileZmaskFixup, NoteDepthControl, WorkerDispatchReads, WorkerDispatchWrites, WorkerDispatchNotPlain, WorkerDispatchLate, WorkerDispatchCopy, Count };
enum class FrameKept : std::uint32_t { NotLatched, AbPhase, SyncFlip, TimingLog, Predicated, LabelForm, DeferredLabels, FullLoopSegmentEnd, FullLoopSettle, FrontChange, Exception, IndirectGpu, IndexInconsistent, DmaCopy, DmaGds, DmaRefused, ResolveLimit, ExactUnsure, WorkerBehindFlip, SuspendNotLatched, SuspendTimingLog, SuspendDrainEnv, SuspendDeferredLabels, SuspendUnordered, IndirectNotLatched, IndirectPredicated, IndirectNoAddress, IndirectDeferredLabels, IndirectItemsOff, IndirectNotCandidate, IndirectNotPlain, IndirectUnbounded, IndirectFallback, IndirectNote, IndirectArgumentFallback, HeldNoDecode, HeldSize, HeldUnbounded, HeldScratch, HeldBda, HeldHeap, HeldWalkFailed, HeldWalkIncomplete, HeldLimit, WorkerBehindIndirect, HeldNotPrepared, HeldTableImage, HeldQueuedWrite, HeldGpuWrite, HeldForcedSync, HeldIndirectWord, Count };
enum class FrameSite : std::uint32_t { OrderedFlip, RenderingWait, SegmentEnd, ExitWait, ExitFree, ArgumentWait, WriteScan, KeyScan, QueuedScan, DmaBeside, ListWait, ResolvedSkip, ResolveWait, SuspendOrdered, IndirectQueued, IndirectArgumentWait, HeldNamed, ExitOwned, Count };
enum class FrameCheck : std::uint32_t { FlipOrder, ItemOrder, Lifetime, FlipState, DispatchArguments, FallbackScan, SegmentEnd, AbPhase, ListBytes, DmaOrder, SuspendOrder, SuspendCompletion, IndirectOrder, IndirectArguments, HeldWrites, Count };
enum class FrameTail : std::uint32_t { RenderingWait, RewindTail, NoSubmission, Count };

class DrawPipeline {
public:
    using Commit = std::function<void()>;
    using Range = std::pair<std::uint64_t, std::uint64_t>;
    struct Watch {
        std::uint64_t retired = 0;
        std::size_t items = 0;
        std::vector<Range> writes;
        bool snapshot = false;
        std::uint32_t snapshotCount = 0;
        std::array<Range, PublishedRanges::Capacity> snapshotWrites{};
        std::uint64_t enqueued = 0;
        bool deep = false;
        bool verified = false;
        std::vector<Range> verifyWrites;
    };
    enum class DrainReason : std::uint8_t { Packet, Flush, Labels, Capture, Indirect, Submission, Count };
    enum class QueuedWrite : std::uint8_t { None, Stored, Unknown };
    enum class CommitKind : std::uint8_t { Other, Draw, IndirectDraw, ColorMetadata, Dispatch, SampleDump, DmaFill, DmaWrite, DmaCopy, Store, EndOfPipe, Seal, SubmitEnd, Completion, BufferFill, Flip, Suspend, IndirectDispatch, Count };

    static DrawPipeline& Queue0();
    static std::size_t Depth();
    static std::size_t QueuedItems();
    static bool& Active();
    static std::atomic<std::uint64_t>& EpochToken();
    static void FollowEpoch(std::uint64_t token);
    static void NoteEpochSince(std::uint64_t token, std::uint64_t since);
    static unsigned Committers();

    using Ahead = void (*)(const void* object, std::uint32_t level);
    void Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress = 0, std::vector<std::byte> labelBytes = {}, std::vector<Range> reads = {}, bool draw = false, Ahead ahead = nullptr, const void* aheadObject = nullptr);
    void Enqueue(CommitKind kind, Commit commit, std::vector<Range> writes, std::uint64_t labelAddress = 0, std::vector<std::byte> labelBytes = {}, std::vector<Range> reads = {}, bool draw = false, Ahead ahead = nullptr, const void* aheadObject = nullptr);
    void EnqueueStore(Commit commit, std::uint64_t address, std::vector<std::byte> bytes);
    void EnqueueLabelRun(Commit commit, std::vector<Range> writes, std::shared_ptr<const LabelRun> run);
    void EnqueueQuiet(Commit commit);
    void EnqueueKept(Commit commit, std::vector<Range> writes, std::vector<Range> reads, std::shared_ptr<const void> keep);
    void EnqueueFlip(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress, std::vector<std::byte> labelBytes);
    void EnqueueSuspend(Commit commit);
    std::uint32_t DescribeQueuedWriter(std::uint64_t begin, std::uint64_t end);
    class FlipCommitScope {
    public:
        FlipCommitScope();
        ~FlipCommitScope();
        FlipCommitScope(const FlipCommitScope&) = delete;
        FlipCommitScope& operator=(const FlipCommitScope&) = delete;

    private:
        bool previous;
    };
    class OrderProbe {
    public:
        explicit OrderProbe(FrameCheck check);
        ~OrderProbe();
        OrderProbe(const OrderProbe&) = delete;
        OrderProbe& operator=(const OrderProbe&) = delete;

    private:
        std::uint32_t previous;
    };
    static void NoteDrawKey(std::uint64_t key, std::uint64_t shape);
    void Drain(DrainReason reason, std::uint32_t opcode = 0x100);
    void AwaitWriters(std::span<const Range> ranges, DrainReason reason, std::uint32_t opcode, DeepSite site);
    bool AwaitExactWriters(std::span<const Range> ranges, DrainReason reason, std::uint32_t opcode, DeepSite site, FrameSite timed, FrameCheck check);
    bool DeepQueue() const { return deepQueue.load(std::memory_order_acquire); }
    bool CommitRing() const { return commitRing.load(std::memory_order_acquire); }
    enum class RingCount : std::uint32_t { FastPops, LockedPops, MutexPops, Enqueues, RoomWaits, RoomWaitNs, Contention, Sleeps, SleepNs, SlowScans, WaiterWakes, Latches, Unlatches, VerifyPops, VerifyScans, VerifyDiffering, Count };
    static std::uint64_t RingTotal(RingCount count);
    static constexpr std::size_t CommitRingSlots = 1024;
    void Stop();
    void ReleaseRetired();
    static std::uint64_t DrainRequestsByOpcode(std::uint32_t opcode);
    static std::uint64_t DrainRequestsByReason(DrainReason reason);
    bool Busy() const;
    std::size_t Outstanding() const { return outstanding.load(std::memory_order_acquire); }
    bool Overlaps(std::uint64_t address, std::size_t bytes);
    bool OverlapsAny(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges);
    bool QueuedClearForValidation(std::span<const Range> ranges);
    Watch WatchWrites();
    bool WrittenSince(const Watch& watch, const std::vector<Range>& reads);
    std::uint64_t Retired() const { return retiredItems.load(std::memory_order_acquire); }
    std::uint64_t Enqueued() const { return enqueuedItems.load(std::memory_order_acquire); }
    void NoteDecodeDrain() { decodeDrains.fetch_add(1, std::memory_order_relaxed); }
    std::uint64_t DecodeDrains() const { return decodeDrains.load(std::memory_order_relaxed); }
    std::optional<std::uint64_t> PendingLabel(std::uint64_t address, std::size_t bytes);
    static bool HeldReads(std::uint64_t address, std::size_t bytes);
    static std::uint64_t HeldReadQueries();
    void ReleaseRunningReads();
    QueuedWrite StoredBytes(std::uint64_t address, std::size_t bytes, std::span<std::byte> out);

    static bool MetadataPassesPipelined();
    static void CountPipelinedMetadataPass();
    static std::uint64_t PipelinedMetadataPasses();
private:
    friend class DrawFinisher;
    struct Item {
        Commit commit;
        std::vector<Range> writes;
        std::uint64_t labelAddress = 0;
        std::vector<std::byte> labelBytes;
        std::vector<Range> reads;
        bool draw = false;
        bool store = false;
        CommitKind kind = CommitKind::Other;
        bool quiet = false;
        std::uint64_t order = 0;
        std::uint32_t parts = 1;
        std::shared_ptr<const void> keep;
        std::shared_ptr<const LabelRun> labelRun;
        bool counted = true;
        std::uint64_t parked = 0;
        std::uint64_t ringOrdinal = 0;
        Ahead ahead = nullptr;
        const void* aheadObject = nullptr;
    };
    class QueuedView {
    public:
        explicit QueuedView(std::deque<Item>& queued) : listed(&queued) {}
        QueuedView(Item* ringSlots, std::uint64_t first, std::uint64_t end) : slots(ringSlots), base(first), count(static_cast<std::size_t>(end - first)) {}
        std::size_t size() const { return listed != nullptr ? listed->size() : count; }
        bool empty() const { return size() == 0; }
        Item& operator[](std::size_t index) const { return listed != nullptr ? (*listed)[index] : slots[static_cast<std::size_t>((base + index) & (CommitRingSlots - 1))]; }
        Item& front() const { return (*this)[0]; }

    private:
        std::deque<Item>* listed = nullptr;
        Item* slots = nullptr;
        std::uint64_t base = 0;
        std::size_t count = 0;
    };
    class QueueScan;
    class RingLock;
    DrawPipeline() = default;
    void enqueue(Item item);
    void queueItem(Item item);
    void verifyOrder(std::uint64_t order);
    void run();
    void rethrowFailure();
    void report(std::chrono::steady_clock::time_point now);
    void countWrites(const std::vector<Range>& writes, bool added);
    bool awaitRetired(std::uint64_t item);
    void latchDeepQueue(bool deep);
    void drainAll(DrainReason reason, std::uint32_t opcode);
    bool queuedOverlap(std::span<const Range> ranges);
    DeepFound deepQueued(std::span<const Range> ranges, std::uint64_t live) const;
    bool deepQueuedWithin(std::span<const Range> ranges, std::uint64_t from, std::uint64_t to) const;
    DeepFound newestQueued(const QueuedView& queued, std::uint64_t retiredNow, std::span<const Range> ranges) const;
    DeepFound scanQueued(std::span<const Range> ranges);
    void verifyExactBytes(std::span<const Range> ranges, DrainReason reason, std::uint32_t opcode, FrameCheck check);
    void verifyQueued(std::uint32_t lookup, std::span<const Range> ranges);
    void noteDeepReads(const std::vector<Range>& reads, std::uint64_t stamp);
    bool readsMaybe(std::uint64_t address, std::size_t bytes) const;
    void verifyHeldReads(std::uint64_t address, std::size_t bytes);
    bool queueLocked(Item& item);
    bool ringEnqueue(Item& item);
    void startCommitter();
    void latchRing();
    void unlatchRing();
    void reclaimRing(std::uint64_t upTo, std::vector<Item>& out);
    Item& ringSlot(std::uint64_t ordinal);
    bool ringEmpty() const;
    bool queueEmptyLocked();
    void lowerRingWake(std::uint64_t target);
    void awaitRingRoom(std::uint64_t resumeFill);
    std::size_t roomLimitNow() const;
    void raiseReadsLive(std::uint64_t live);
    void ringReport();
    std::string ringText();

    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    std::deque<Item> items;
    std::vector<std::pair<Range, std::uint32_t>> queuedWrites;
    PublishedRanges published;
    std::atomic<std::uint64_t> enqueuedItems{0};
    std::vector<Item> retired;
    std::exception_ptr failure;
    std::atomic<std::size_t> outstanding{0};
    std::atomic<std::uint64_t> retiredItems{0};
    std::atomic<std::uint64_t> decodeDrains{0};
    std::uint64_t reportedDecodeDrains = 0;
    std::uint64_t reportedDeposits = 0;
    std::uint64_t reportedTurnItems = 0;
    std::uint64_t verifiedOrder = 0;
    std::uint64_t orderOrdinal = 0;
    std::uint32_t idleWaiters = 0;
    std::uint32_t retireWaiters = 0;
    std::atomic<bool> deepQueue{false};
    static constexpr std::size_t ReadRegionSlots = 4096;
    std::unique_ptr<StampedRanges> deepStores;
    std::unique_ptr<StampedRanges> deepOthers;
    std::unique_ptr<std::array<std::atomic<std::uint64_t>, ReadRegionSlots>> readRegions;
    std::atomic<bool> deepIndexed{false};
    std::atomic<std::uint64_t> wideReads{0};
    std::atomic<std::uint64_t> deepEpoch{0};
    alignas(64) std::atomic<std::uint64_t> readsLive{1};
    alignas(64) std::atomic<bool> commitRing{false};
    std::atomic<bool> ringClosing{false};
    std::atomic<bool> failurePending{false};
    std::atomic<bool> committerSleeping{false};
    std::atomic<std::uint32_t> ringScans{0};
    std::atomic<std::uint32_t> ringIdleWaiters{0};
    std::atomic<std::uint64_t> ringWakeAt{~std::uint64_t{0}};
    std::atomic<std::int64_t> ringReportAt{0};
    std::unique_ptr<std::array<Item, CommitRingSlots>> ring;
    alignas(64) std::atomic<bool> ringLocked{false};
    std::uint64_t ringReclaimed = 0;
    alignas(64) std::atomic<bool> committerRetiring{false};
    std::atomic<std::uint64_t> ringCommits{0};
    std::atomic<std::uint64_t> ringCommitNs{0};
    std::atomic<std::uint64_t> ringCommitErrors{0};
    std::uint64_t reportedRingCommits = 0;
    std::uint64_t reportedRingCommitNs = 0;
    std::uint64_t reportedRingCommitErrors = 0;
    std::size_t resumeAt = 0;
    bool committerWaiting = false;
    bool stopped = false;
    bool closing = false;
    std::uint64_t wakes = 0;
    std::uint64_t lockWaitNs = 0;
    std::thread thread;
    std::uint64_t commits = 0;
    std::uint64_t commitNs = 0;
    std::uint64_t commitErrors = 0;
    std::uint64_t enqueued = 0;
    std::uint64_t fullWaitNs = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrainReason::Count)> drains{};
    std::array<std::uint64_t, static_cast<std::size_t>(DrainReason::Count)> drainWaitNs{};
    std::array<std::uint64_t, 257> drainOpcodes{};
    std::array<std::atomic<std::uint64_t>, 257> drainRequests{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrainReason::Count)> reasonRequests{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(CommitKind::Count)> kindCommits{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(CommitKind::Count)> kindNs{};
    alignas(64) std::uint64_t depthSum = 0;
    std::size_t deepestItems = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

class DrawFinisher {
public:
    using Job = std::function<void()>;
    using Range = std::pair<std::uint64_t, std::uint64_t>;
    using LeanRunner = void (*)(const std::byte* payload);
    using LeanRetirer = void (*)(const std::byte* payload, std::shared_ptr<const void>&& answer);
    static constexpr std::size_t LeanPayloadBytes = 1024;
    static constexpr std::size_t LeanSlots = 1024;
    static constexpr std::size_t NarrowLeanSlots = 64;
    class AheadScope {
    public:
        AheadScope();
        ~AheadScope();
        AheadScope(const AheadScope&) = delete;
        AheadScope& operator=(const AheadScope&) = delete;

    private:
        bool previous;
    };
    struct Pending {
        std::uint64_t key = 0;
        std::uint64_t shape = 0;
        bool inserts = true;
        bool unknown = true;
        std::vector<Range> writes;
        std::uint64_t labelAddress = 0;
        std::vector<std::byte> labelBytes;
        std::shared_ptr<const LabelRun> labelRun;
        bool named = false;
        std::uint8_t origin = 0;
        std::uint8_t originKind = 0;
    };
    enum class LabelLookup : std::uint8_t { NotHeld, Value, Unknown };
    struct DepositCounts {
        std::uint64_t deposited = 0;
        std::uint64_t forwarded = 0;
        std::uint64_t direct = 0;
    };
    static DepositCounts Deposits();
    static bool Overlap();
    bool PendingOverlaps(std::uint64_t begin, std::uint64_t end);
    std::uint64_t ParkReads(const std::vector<Range>& reads);
    void UnparkReads(std::uint64_t id);
    bool ParkedReadsOverlap(std::uint64_t address, std::size_t bytes);
    LabelLookup PendingLabel(std::uint64_t address, std::size_t bytes, std::uint64_t& value);
    bool PendingLabelBytes(std::uint64_t address, std::span<std::byte> out);
    bool PendingLabelSpan(std::uint64_t address, std::size_t bytes);
    bool PendingOverlapsAny(std::span<const Range> ranges);
    bool PendingClearForValidation(std::span<const Range> ranges);
    static bool UnknownWritesAhead();
    bool PendingKey(std::uint64_t key, std::uint64_t shape);
    enum class DrainSite : std::uint32_t { Lookup, Reads, Packet, Kept, Count };
    static void NoteDrain(DrainSite site, bool drained);
    enum class FlowSite : std::uint32_t { PacketFinisher, PacketCommitter, Predication, Dispatch, DispatchReads, Indirect, Relocation, Capture, Count };
    enum class FlowLookup : std::uint32_t { SameKey, EarlierKey, PendingWrites, QueuedWrites, HeldReads, Watch, Count };
    static bool Flow();
    static bool FlowSkips(FlowSite site);
    static void NoteFlowLookup(FlowLookup lookup);
    static std::string FlowText();
    static bool Deep();
    bool DeepRing() const { return deepRing.load(std::memory_order_acquire); }
    void AwaitKeyed(std::uint64_t key, std::uint64_t shape, bool insertsOnly);
    void AwaitPending(std::span<const Range> ranges, DeepSite site);
    static void NoteDeepKept(DeepKept kept, std::uint32_t opcode = 0x100);
    static void NoteDeepSite(DeepSite site);
    static std::string DeepText();
    enum class DeepCount : std::uint32_t { WriteOverflows, KeyOverflows, QueuedOverflows, WorkerFallbacks, IndexFull, Count };
    static std::uint64_t DeepTotal(DeepCount count);
    std::uint64_t KeyBlocker(std::uint64_t key, std::uint64_t shape, bool insertsOnly);
    static std::string DrainText();
    static void NoteDrainedOpcode(std::uint32_t opcode, bool besideCommits);
    static DrawFinisher& Get();
    static bool Enabled();
    static bool OnFinisherThread();
    static unsigned Finishers();
    static bool Busy();
    using RunFlusher = void (*)();
    void SetOpenRun(RunFlusher flusher) {
        openRun = flusher;
        runOpen.store(flusher != nullptr, std::memory_order_release);
    }
    void FlushOpenRun();
    void HoldOpenRunWrite(const Range& range);
    std::uint64_t Issued() const { return issued.load(std::memory_order_relaxed); }
    std::uint64_t Front() const { return frontSeq.load(std::memory_order_acquire); }
    std::size_t SlotIndex(std::uint64_t seq) const { return static_cast<std::size_t>(seq & leanMask.load(std::memory_order_relaxed)); }
    unsigned IdleState() const;
    void Enqueue(Job job);
    void Enqueue(Job job, Pending held);
    void Drain(bool helping = false);
    void Settle();
    void Stop();
    void ReleaseRan();
    void AwaitTurn();
    bool Adopt(bool ring);
    bool RingStore() const { return ringMode.load(std::memory_order_acquire); }
    std::byte* ReserveLean();
    void PublishLean(LeanRunner runner, std::span<const Range> writes, std::uint64_t key, std::uint64_t shape, LeanRetirer retirer = nullptr, bool batch = false);
    static bool AnswerLean(std::shared_ptr<const void> answer);
    PublishedKeys::Answer KeyBefore(std::uint64_t key, std::uint64_t shape, bool insertsOnly) const;
    bool AheadBusy() const;
    static bool AheadOfTurn();
    static bool DeferRelease(std::array<std::shared_ptr<const void>, 3>& held) noexcept;
    static std::uint64_t TakeDeepestRelease();
    static bool FrameFlowWanted();
    static bool FrameFlow();
    static bool OrderedSyncsWanted();
    static bool& NamingWalk();
    enum class NamingRefused : std::uint8_t { None, QueuedWrite, GpuWrite };
    static NamingRefused& NamingRefusal();
    static std::string& NamingTrace();
    static void NoteNamingBlocker(std::uint64_t begin, std::uint64_t end);
    static void NoteEvidenceWait(std::uint64_t begin, std::uint64_t end, bool buffered);
    std::uint32_t describePendingWriter(std::uint64_t begin, std::uint64_t end);
    static bool OrderedSyncs();
    static void NoteHeldNaming(bool named, std::size_t ranges, std::uint64_t ns);
    static void NoteHeldWalk(bool named, FrameKept reason, std::uint64_t ns);
    static std::uint64_t HeldNamedTotal();
    static std::uint64_t HeldUnknownTotal();
    static bool FrameCensus();
    bool FrameLatch() const { return frameLatch.load(std::memory_order_acquire); }
    void LatchFrameFlow();
    static void NoteFrameKept(FrameKept kept);
    static void NoteFrameSite(FrameSite site, std::uint64_t ns = 0);
    static void NoteFlipDistance(std::uint64_t jobs, std::uint64_t items, std::uint64_t inFlight);
    static std::size_t OutstandingJobs() { return outstanding.load(std::memory_order_acquire); }
    void NoteFlipQueued() { flipsQueued.fetch_add(1, std::memory_order_acq_rel); }
    void NoteFlipSeq(std::uint64_t seq) { lastFlipSeq.store(seq, std::memory_order_release); }
    void NoteFlipCommitted() { flipsCommitted.fetch_add(1, std::memory_order_acq_rel); }
    std::uint64_t FlipsQueued() const { return flipsQueued.load(std::memory_order_acquire); }
    std::uint64_t FlipsCommitted() const { return flipsCommitted.load(std::memory_order_acquire); }
    std::uint64_t LastFlipSeq() const { return lastFlipSeq.load(std::memory_order_acquire); }
    void NoteIndirectQueued() { indirectsQueued.fetch_add(1, std::memory_order_acq_rel); }
    void NoteIndirectCommitted() { indirectsCommitted.fetch_add(1, std::memory_order_acq_rel); }
    bool IndirectsInFlight() const { return indirectsQueued.load(std::memory_order_acquire) != indirectsCommitted.load(std::memory_order_acquire); }
    void NoteRawReader() { rawReader = nextSeq - 1; }
    std::uint64_t RawReader() const { return rawReader; }
    bool AwaitRawReaders();
    void AwaitExactPending(std::span<const Range> ranges, DeepSite site, FrameSite timed);
    static std::uint64_t FrameVerifyEvery();
    static bool FrameVerifyDue(FrameCheck check);
    static void NoteFrameCheck(FrameCheck check, bool differs, const std::string& detail);
    static std::string FrameText();
    static void NoteFlipJobs(std::uint64_t jobs);
    static void RetireFlipJobs();
    class RawReaderHold {
    public:
        explicit RawReaderHold(const void* submission);
        ~RawReaderHold();
        RawReaderHold(const RawReaderHold&) = delete;
        RawReaderHold& operator=(const RawReaderHold&) = delete;
        void Release();

    private:
        const void* submission;
        std::atomic<bool> held{true};
    };
    static std::shared_ptr<RawReaderHold> HoldRawReader(const void* submission);
    static void CheckRawReaders(const void* submission, const char* where);
    class TailScope {
    public:
        explicit TailScope(FrameTail cause);
        ~TailScope();
        TailScope(const TailScope&) = delete;
        TailScope& operator=(const TailScope&) = delete;

    private:
        std::chrono::steady_clock::time_point start{};
        std::uint32_t reason = 0;
    };

private:
    friend class DrawPipeline;
    struct Slot {
        Job job;
        std::uint64_t seq = 0;
        bool taken = false;
        bool done = false;
    };
    struct Deposit {
        DrawPipeline* pipeline = nullptr;
        DrawPipeline::Item item;
    };
    struct LeanSlot {
        std::atomic<std::uint64_t> seq{0};
        std::atomic<std::uint8_t> state{0};
        LeanRunner runner = nullptr;
        LeanRetirer retirer = nullptr;
        bool batch = false;
        bool counted = true;
        std::uint64_t unknownBefore = 0;
        std::atomic<std::uint8_t> resolved{0};
        std::vector<Range> resolvedWrites;
        std::shared_ptr<const void> answer;
        Job closure;
        Pending pending;
        std::vector<Deposit> deposits;
        alignas(16) std::array<std::byte, LeanPayloadBytes> payload{};
    };
    struct ParkSlot {
        std::mutex mutex;
        std::condition_variable wake;
        bool signaled = false;
    };
    DrawFinisher() = default;
    void run(unsigned index);
    Slot* untaken();
    bool countPending(const Pending& held, bool added);
    void startFinishers();
    LeanSlot& leanSlot(std::uint64_t seq) { return (*lean)[SlotIndex(seq)]; }
    bool leanReady() const { return leanTake.load(std::memory_order_acquire) < leanTail.load(std::memory_order_acquire); }
    bool takeLean(unsigned index, std::uint64_t& seq);
    void runLean(std::uint64_t seq, std::chrono::steady_clock::time_point& lastEnd);
    std::uint64_t advanceLeanFront();
    void awaitTurn(std::uint64_t seq);
    void awaitFrontPast(std::uint64_t job);
    bool deposit(DrawPipeline& pipeline, DrawPipeline::Item& item);
    void forwardOwn(std::uint64_t seq);
    bool forwardLean(LeanSlot& slot, std::uint64_t seq, std::uint64_t& forwardNs);
    static void forwardMerged(std::vector<Deposit>& deposits, std::size_t& sent);
    const std::vector<Deposit>& ownDeposits();
    bool depositsWrite(std::span<const Range> ranges);
    void verifyNamedWrites(const DrawPipeline::Item& item);
    bool depositsRead(std::uint64_t address, std::size_t bytes);
    void checkDrainDeposits();
    void retireLean();
    void rethrowLeanFailure();
    void fitLeanSpan();
    void latchDeepRing(bool deep);
    DeepAnswer deepPendingHit(std::span<const Range> ranges);
    DeepFound keysBefore(std::uint64_t key, std::uint64_t shape, bool insertsOnly, std::uint64_t live, std::uint64_t before) const;
    DeepFound scanWrites(std::span<const Range> ranges, std::uint64_t from, std::uint64_t to) const;
    DeepFound scanKeys(std::uint64_t key, std::uint64_t shape, bool insertsOnly, std::uint64_t from, std::uint64_t to) const;
    bool awaitProducerFront(std::uint64_t job);
    void awaitLeanRoom(bool helping = false);
    void awaitLeanIdle();
    bool takeHelped(std::uint64_t& seq);
    template<typename TWaiting>
    void helpLean(TWaiting&& waiting, bool room);
    void publishLean(LeanSlot& slot);
    void signalSlots(std::uint32_t bits);
    void wakeParked(std::uint32_t slotted, unsigned wanted);
    bool parkInSlot(unsigned index);
    bool releaseRetiredHits(bool stopForJobs);
    void releaseAllRetiredHits();
    void verifiedIdleWait(std::unique_lock<std::mutex>& lock, const std::function<bool()>& ready);
    bool aheadWait(std::span<const Range> ranges);
    struct ExactFound {
        bool unsure = false;
        bool limited = false;
        std::uint64_t blocker = 0;
        std::uint64_t unresolved = 0;
        std::uint64_t skipped = 0;
    };
    ExactFound exactPending(std::span<const Range> ranges);
    bool awaitResolved(std::uint64_t job);
    std::uint64_t scanBlocker(std::span<const Range> ranges, FrameSite site);
    std::uint64_t keyScanBlocker(std::uint64_t key, std::uint64_t shape, bool insertsOnly);
    void verifyScanned(std::span<const Range> ranges, std::uint64_t blocker);
    inline static std::atomic<std::size_t> outstanding{0};
    inline static std::atomic<bool> runOpen{false};
    RunFlusher openRun = nullptr;
    std::atomic<std::size_t> untakenJobs{0};
    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    std::deque<Slot> jobs;
    std::deque<Pending> pending;
    std::vector<std::pair<Range, std::uint32_t>> pendingWrites;
    std::uint32_t pendingUnknown = 0;
    std::vector<std::uint64_t> unknownSeqs;
    PublishedRanges published;
    PublishedKeys keys;
    std::vector<Job> ran;
    std::exception_ptr failure;
    bool running = false;
    std::array<bool, 8> extraRunning{};
    unsigned sleepers = 0;
    unsigned turnSleepers = 0;
    unsigned idleWaiters = 0;
    std::atomic<std::uint64_t> frontSeq{1};
    std::atomic<std::uint64_t> issued{1};
    bool closing = false;
    std::uint64_t nextSeq = 1;
    std::condition_variable turnWake;
    std::condition_variable dormantWake;
    std::thread thread;
    std::array<std::thread, 8> extra;
    std::atomic<bool> ringMode{false};
    std::atomic<bool> closed{false};
    std::atomic<bool> leanFailed{false};
    std::atomic<unsigned> started{0};
    std::atomic<unsigned> parked{0};
    std::atomic<unsigned> dormant{0};
    std::atomic<unsigned> frontWaiters{0};
    std::atomic<std::uint32_t> parkedSlots{0};
    std::atomic<std::uint32_t> idleSpinners{0};
    std::atomic<std::uint32_t> wakesInFlight{0};
    std::array<ParkSlot, 9> parkSlots;
    std::atomic<std::uint64_t> leanTake{1};
    std::atomic<std::uint64_t> leanTail{1};
    std::atomic<std::uint64_t> leanMask{NarrowLeanSlots - 1};
    std::atomic<bool> deepRing{false};
    std::unique_ptr<StampedRanges> deepWrites;
    std::unique_ptr<StampedKeys> deepKeys;
    std::unique_ptr<StampedKeys> deepShapes;
    std::atomic<bool> deepIndexed{false};
    std::atomic<std::uint64_t> lastUnknown{0};
    std::atomic<bool> frameLatch{false};
    std::atomic<std::uint64_t> frameGeneration{0};
    std::uint64_t rawReader = 0;
    std::atomic<std::uint64_t> flipsQueued{0};
    std::atomic<std::uint64_t> flipsCommitted{0};
    std::atomic<std::uint64_t> lastFlipSeq{0};
    std::atomic<std::uint64_t> indirectsQueued{0};
    std::atomic<std::uint64_t> indirectsCommitted{0};
    std::atomic<std::uint32_t> resolveWaiters{0};
    std::vector<Range> openRunWrites;
    mutable std::mutex verifySlots;
    std::uint64_t leanRetired = 1;
    std::unique_ptr<std::array<LeanSlot, LeanSlots>> lean;
    std::mutex parkedReadsMutex;
    std::vector<std::pair<std::uint64_t, std::vector<Range>>> parkedReads;
    std::uint64_t nextParkedReads = 1;
    std::atomic<std::uint32_t> parkedReadCount{0};
};

namespace LabelRunCensus {

enum class Gap : std::uint8_t { First, Adjacent, Registers, Waits, Draws, Dispatches, Other, Far, Count };
enum class End : std::uint8_t { SegmentEnd, Draw, Dispatch, Wait, Registers, Other, Unqueued, Interrupt, Check, Cap, Count };

bool Enabled();
void NoteQueued(Gap gap, std::uint32_t otherOpcode, bool item);
void NoteRun(std::size_t packets, std::size_t items, End end, std::uint32_t otherOpcode);
void NoteVerify(std::size_t packets, std::size_t differing);
void NoteCommitted(DrawPipeline::CommitKind previous, bool first, std::uint32_t labels);
std::string Take();

}

}

#endif
