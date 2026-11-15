#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWTICKETS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWTICKETS_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRecords.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <vector>

namespace AgcDriver::DriverDetail {

struct LeanDrawJob;

enum class TicketFallback : std::uint8_t { NoArena, Pooled, Room, Size, Device, Owners, Identity, Verify, Count };

class TicketArena;

struct TicketCold {
    std::vector<std::byte> blob;
    std::shared_ptr<Graphics::ShaderResources::DrawBindings> placed;
    std::shared_ptr<const void> owner;
    std::shared_ptr<const DrawDecode> decode;
};

struct alignas(64) DrawTicket {
    static constexpr std::uint32_t Buffers = 1;
    static constexpr std::uint32_t Inputs = 2;
    static constexpr std::uint32_t Fragment = 4;
    static constexpr std::uint32_t PlacedHeld = 8;

    TicketArena* arena = nullptr;
    std::uint64_t seq = 0;
    std::atomic<std::uint32_t> abandoned{0};
    std::uint32_t bytes = 0;
    std::uint32_t followerBytes = 0;
    std::uint32_t parts = 0;
    std::uint16_t stageCount = 0;
    std::uint16_t sideCount = 0;
    std::uint16_t captureCount = 0;
    std::uint16_t movedCount = 0;
    std::uint16_t selectedCount = 0;
    std::uint16_t rangeCount = 0;
    std::uint16_t pairCount = 0;
    std::uint16_t handleCount = 0;
    std::uint16_t offsetCount = 0;
    std::uint16_t readCount = 0;
    std::uint16_t ownerCount = 0;
    std::uint32_t rangesAt = 0;
    std::uint32_t pairsAt = 0;
    std::uint32_t movedAt = 0;
    std::uint32_t selectedAt = 0;
    std::uint32_t stagesAt = 0;
    std::uint32_t coldAt = 0;
    std::uint32_t readsAt = 0;
    std::uint32_t handlesAt = 0;
    std::uint32_t offsetsAt = 0;
    std::uint32_t capturesAt = 0;
    std::uint32_t sidesAt = 0;
    DrawRecordCommit* commit = nullptr;
    VulkanDevice* device = nullptr;
    const Graphics::State* state = nullptr;
    std::uint32_t queue = 0;
    std::uint32_t opcode = 0;
    std::uint64_t epoch = 0;
    std::uint64_t shape = 0;
    Graphics::DrawTraceContext trace{};
    Pm4::DrawParameters parameters{};
    std::shared_ptr<const Graphics::ShaderResources::BufferPlan> plan;
    const std::byte* blobData = nullptr;
    std::size_t blobBytes = 0;
    std::uint64_t generation = 0;
    std::uint64_t depthGuards = 0;
    std::uint64_t bufferForget = 0;
    std::uint64_t bufferImports = 0;
    std::uint64_t bufferMapping = 0;
    VkDevice placedDevice = VK_NULL_HANDLE;
    std::uint32_t lookupsDeferred = 0;
    std::uint32_t lookupsForced = 0;
    Graphics::ShaderResources::PreparedBuffers::Refusal bufferRefusal = Graphics::ShaderResources::PreparedBuffers::Refusal::None;
    std::uint8_t shortcuts = 0;
    bool inPlace = false;
    bool aliasesClear = false;
    std::shared_ptr<const DrawRecipe> recipe;
    const Graphics::State* inputState = nullptr;
    VkDevice inputDevice = VK_NULL_HANDLE;
    std::uint64_t inputForget = 0;
    std::uint64_t inputImports = 0;
    std::uint64_t inputMapping = 0;
    std::uint64_t indexBytes = 0;
    std::size_t indirectRanges = 0;
    std::uint32_t maxIndex = 0;
    Graphics::PreparedInputs::Refusal inputRefusal = Graphics::PreparedInputs::Refusal::None;
    bool exactCapacity = false;
    DrawFragment fragment;
    std::array<Graphics::KeptOwner, 2> owners{};

    template <typename T>
    T* At(std::uint32_t offset) noexcept {
        return std::launder(reinterpret_cast<T*>(reinterpret_cast<std::byte*>(this) + offset));
    }
    template <typename T>
    const T* At(std::uint32_t offset) const noexcept {
        return std::launder(reinterpret_cast<const T*>(reinterpret_cast<const std::byte*>(this) + offset));
    }
    bool Has(std::uint32_t part) const noexcept { return (parts & part) != 0; }
    std::span<const Graphics::CompiledShader> Stages() const noexcept {
        if (stageCount == 0) return {};
        return {At<Graphics::CompiledShader>(stagesAt), stageCount};
    }
    std::span<const DrawPipeline::Range> Reads() const noexcept {
        if (readCount == 0) return {};
        return {At<DrawPipeline::Range>(readsAt), readCount};
    }
    std::span<const Graphics::KeptOwner> Owners() const noexcept { return {owners.data(), ownerCount}; }
    TicketCold& Cold() noexcept { return *At<TicketCold>(coldAt); }
    Graphics::ShaderResources::PreparedBuffersView BuffersView() const noexcept;
    Graphics::PreparedInputsView InputsView() const noexcept;
};

class TicketArena {
public:
    static constexpr std::size_t RingBytes = std::size_t{64} << 10;
    static constexpr std::size_t MaxTicketBytes = std::size_t{16} << 10;
    static constexpr std::size_t SideStack = 64;
    static constexpr std::size_t BlobStack = 32;

    static TicketArena* Current() noexcept;
    static TicketArena* ForThread() noexcept;
    static void BindThread(unsigned index) noexcept;
    static void Orphan() noexcept;
    static void ReclaimOrphans() noexcept;
    static void ReclaimOrphansAtIdle() noexcept;
    static bool OrphansWaiting() noexcept;
    bool ReclaimOne() noexcept;
    void PublishRetired(std::uint64_t seq) noexcept;

    void* Allocate(std::size_t bytes, const std::shared_ptr<VulkanDevice>& device, TicketFallback& refusal) noexcept;
    std::uint64_t NextSeq() noexcept { return ++issued; }
    const std::shared_ptr<VulkanDevice>& HeldDevice() const noexcept { return heldDevice; }
    void RefillBlob(std::vector<std::byte>& blob) noexcept;
    void RefillSide(ShaderRecompiler::HeapStageWords& side) noexcept;

    TicketArena();
    ~TicketArena();
    TicketArena(const TicketArena&) = delete;
    TicketArena& operator=(const TicketArena&) = delete;

private:
    static void sweepOrphans() noexcept;
    std::size_t place(std::size_t bytes) noexcept;
    bool reclaimTail(bool idle) noexcept;
    void destroy(DrawTicket& ticket) noexcept;

    alignas(64) std::atomic<std::uint64_t> retired{0};
    alignas(64) std::unique_ptr<std::byte[]> storage;
    std::byte* ring = nullptr;
    std::size_t head = 0;
    std::size_t tail = 0;
    std::size_t used = 0;
    std::size_t wrapEnd = RingBytes;
    std::uint64_t issued = 0;
    std::shared_ptr<VulkanDevice> heldDevice;
    std::vector<std::vector<std::byte>> blobs;
    std::vector<ShaderRecompiler::HeapStageWords> sides;
    bool orphaned = false;
};

struct TicketRetire {
    TicketArena* arena = nullptr;
    std::uint64_t seq = 0;
};

TicketRetire RetireOf(const void* ticket) noexcept;
void AbandonTicket(void* ticket) noexcept;

DrawTicket* FlattenTicket(TicketArena& arena, DrawRecordHit& draft, const LeanDrawJob& job, const std::shared_ptr<VulkanDevice>& device, TicketFallback& refusal);
std::span<const Graphics::KeptOwner> FollowerOwners(const DrawTicket& ticket, std::array<Graphics::KeptOwner, 2>& recomputed);
void RefillTicketSide(ShaderRecompiler::HeapStageWords& side) noexcept;

class ThawedTicket {
public:
    explicit ThawedTicket(DrawTicket& ticket);
    ~ThawedTicket();
    ThawedTicket(const ThawedTicket&) = delete;
    ThawedTicket& operator=(const ThawedTicket&) = delete;
    DrawRecordHit& Hit() noexcept { return *hit; }
    std::shared_ptr<VulkanDevice> TakeDevice() noexcept { return std::move(hit->device); }

private:
    DrawTicket& ticket;
    DrawRecordHit* hit = nullptr;
    std::unique_ptr<DrawRecordHit> nested;
    bool* busy = nullptr;
    bool timed = false;
};

namespace DrawTickets {

enum class Counter : std::uint8_t { Eligible, Made, Committed, Followed, PrefetchedFollowers, Thaws, ThawNanoseconds, ThawsTimed, Bytes, FollowerBytes, Owners, ReclaimedIdle, ReclaimedInJob, Abandoned, VerifyChecked, VerifyDiffering, ThawAllocations, ThawsCounted, Count };

void Add(Counter counter, std::uint64_t count = 1);
void AddFallback(TicketFallback reason);
void NoteCommit(bool followed, bool prefetched);
std::uint64_t Total(Counter counter);
void Report();

}

}

#endif
