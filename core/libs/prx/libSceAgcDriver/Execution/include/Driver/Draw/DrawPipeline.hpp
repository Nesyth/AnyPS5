#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWPIPELINE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWPIPELINE_HPP

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
    enum class Answer : std::uint8_t { Clear, Hit, Unsure };
    void Publish(std::span<const std::pair<Range, std::uint32_t>> entries, std::uint32_t unknownHeld);
    Answer Overlaps(std::span<const Range> ranges) const;
    bool Copy(std::array<Range, Capacity>& out, std::uint32_t& size) const;

private:
    std::atomic<std::uint32_t> seq{0};
    std::atomic<std::uint32_t> count{0};
    std::atomic<std::uint32_t> unknown{0};
    std::atomic<std::uint64_t> lowest{0};
    std::atomic<std::uint64_t> highest{0};
    std::array<std::atomic<std::uint64_t>, Capacity * 2> bounds{};
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
    };
    enum class DrainReason : std::uint8_t { Packet, Flush, Labels, Capture, Indirect, Submission, Count };
    enum class QueuedWrite : std::uint8_t { None, Stored, Unknown };
    enum class CommitKind : std::uint8_t { Other, Draw, IndirectDraw, ColorMetadata, Dispatch, SampleDump, DmaFill, DmaWrite, DmaCopy, Store, EndOfPipe, Seal, SubmitEnd, Completion, BufferFill, Count };

    static DrawPipeline& Queue0();
    static std::size_t Depth();
    static std::size_t QueuedItems();
    static bool& Active();
    static std::atomic<std::uint64_t>& EpochToken();
    static void FollowEpoch(std::uint64_t token);
    static unsigned Committers();

    void Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress = 0, std::vector<std::byte> labelBytes = {}, std::vector<Range> reads = {}, bool draw = false);
    void Enqueue(CommitKind kind, Commit commit, std::vector<Range> writes, std::uint64_t labelAddress = 0, std::vector<std::byte> labelBytes = {}, std::vector<Range> reads = {}, bool draw = false);
    void EnqueueStore(Commit commit, std::uint64_t address, std::vector<std::byte> bytes);
    void EnqueueQuiet(Commit commit);
    static void NoteDrawKey(std::uint64_t key, std::uint64_t shape);
    void Drain(DrainReason reason, std::uint32_t opcode = 0x100);
    void Stop();
    void ReleaseRetired();
    static std::uint64_t DrainRequestsByOpcode(std::uint32_t opcode);
    static std::uint64_t DrainRequestsByReason(DrainReason reason);
    bool Busy() const;
    std::size_t Outstanding() const { return outstanding.load(std::memory_order_acquire); }
    bool Overlaps(std::uint64_t address, std::size_t bytes);
    bool OverlapsAny(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges);
    Watch WatchWrites();
    bool WrittenSince(const Watch& watch, const std::vector<Range>& reads);
    std::uint64_t Retired() const { return retiredItems.load(std::memory_order_acquire); }
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
    };
    DrawPipeline() = default;
    void enqueue(Item item);
    void queueItem(Item item);
    void verifyOrder(std::uint64_t order);
    void run();
    void rethrowFailure();
    void report(std::chrono::steady_clock::time_point now);
    void countWrites(const std::vector<Range>& writes, bool added);

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
    std::uint64_t depthSum = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

class DrawFinisher {
public:
    using Job = std::function<void()>;
    using Range = std::pair<std::uint64_t, std::uint64_t>;
    using LeanRunner = void (*)(const std::byte* payload);
    using LeanRetirer = void (*)(const std::byte* payload, std::shared_ptr<const void>&& answer);
    static constexpr std::size_t LeanPayloadBytes = 1024;
    static constexpr std::size_t LeanSlots = 64;
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
    bool PendingOverlapsAny(std::span<const Range> ranges);
    bool PendingKey(std::uint64_t key, std::uint64_t shape);
    enum class DrainSite : std::uint32_t { Lookup, Reads, Packet, Kept, Count };
    static void NoteDrain(DrainSite site, bool drained);
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
    unsigned IdleState() const;
    void Enqueue(Job job);
    void Enqueue(Job job, Pending held);
    void Drain();
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
        std::shared_ptr<const void> answer;
        Job closure;
        Pending pending;
        std::vector<Deposit> deposits;
        alignas(16) std::array<std::byte, LeanPayloadBytes> payload{};
    };
    DrawFinisher() = default;
    void run(unsigned index);
    Slot* untaken();
    bool countPending(const Pending& held, bool added);
    void startFinishers();
    LeanSlot& leanSlot(std::uint64_t seq) { return (*lean)[static_cast<std::size_t>(seq % LeanSlots)]; }
    bool leanReady() const { return leanTake.load(std::memory_order_acquire) < leanTail.load(std::memory_order_acquire); }
    bool takeLean(unsigned index, std::uint64_t& seq);
    void runLean(std::uint64_t seq, std::chrono::steady_clock::time_point& lastEnd);
    std::uint64_t advanceLeanFront();
    void awaitTurn(std::uint64_t seq);
    bool deposit(DrawPipeline& pipeline, DrawPipeline::Item& item);
    void forwardOwn(std::uint64_t seq);
    bool forwardLean(LeanSlot& slot, std::uint64_t seq, std::uint64_t& forwardNs);
    static void forwardMerged(std::vector<Deposit>& deposits, std::size_t& sent);
    const std::vector<Deposit>& ownDeposits();
    bool depositsWrite(std::span<const Range> ranges);
    bool depositsRead(std::uint64_t address, std::size_t bytes);
    void checkDrainDeposits();
    void retireLean();
    void rethrowLeanFailure();
    void awaitLeanRoom();
    void awaitLeanIdle();
    void publishLean(LeanSlot& slot);
    bool aheadWait(std::span<const Range> ranges);
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
    std::atomic<std::uint64_t> leanTake{1};
    std::atomic<std::uint64_t> leanTail{1};
    std::uint64_t leanRetired = 1;
    std::unique_ptr<std::array<LeanSlot, LeanSlots>> lean;
    std::mutex parkedReadsMutex;
    std::vector<std::pair<std::uint64_t, std::vector<Range>>> parkedReads;
    std::uint64_t nextParkedReads = 1;
    std::atomic<std::uint32_t> parkedReadCount{0};
};

}

#endif
