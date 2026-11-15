#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawTickets.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DirectDraws.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/LeanDraw.hpp"
#include "prx/libSceAgcDriver/Execution/include/HostAllocator.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "prx/libc/include/HostThreadSlot.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace AgcDriver::DriverDetail {

namespace {

using Graphics::KeptOwner;
using MovedBuffer = Graphics::ShaderResources::MovedBuffer;
using Selected = Graphics::ShaderResources::PreparedBuffers::Selected;
using InputRange = Graphics::PreparedInputs::Range;
using ReadPair = std::pair<std::uint64_t, std::uint64_t>;
using Capture = std::shared_ptr<const ShaderRecompiler::ResourceCapture>;
using Side = ShaderRecompiler::HeapStageWords;

constexpr std::size_t Line = 64;

constexpr std::size_t alignUp(std::size_t value, std::size_t to) {
    return (value + to - 1) & ~(to - 1);
}

struct ArenaSlotTag {};
struct BoundSlotTag {};
using CurrentArena = HostThreadSlot<TicketArena*, ArenaSlotTag>;
using BoundIndex = HostThreadSlot<std::uint32_t, BoundSlotTag>;

struct OrphanRegistry {
    std::mutex mutex;
    std::vector<TicketArena*> orphans;
};

OrphanRegistry& orphanRegistry() {
    static auto* kept = new OrphanRegistry;
    return *kept;
}

std::atomic<std::uint32_t> orphansWaiting{0};

constexpr std::size_t CounterCount = static_cast<std::size_t>(DrawTickets::Counter::Count);
constexpr std::size_t FallbackCount = static_cast<std::size_t>(TicketFallback::Count);
constexpr std::array<const char*, FallbackCount> FallbackNames{"no arena", "pooled hits", "room", "size", "device", "upload chunk owners", "identity", "verify"};

struct CounterBlock {
    std::array<std::atomic<std::uint64_t>, CounterCount> counters{};
    std::array<std::atomic<std::uint64_t>, FallbackCount> fallbacks{};
    std::atomic<std::uint64_t> maxBytes{0};
    std::atomic<std::uint64_t> maxFollower{0};
    std::atomic<std::uint64_t> highWater{0};
};

struct CounterRegistry {
    std::mutex mutex;
    std::vector<CounterBlock*> live;
    CounterBlock retired;
};

CounterRegistry& counterRegistry() {
    static auto* kept = new CounterRegistry;
    return *kept;
}

void bump(std::atomic<std::uint64_t>& value, std::uint64_t count) {
    value.store(value.load(std::memory_order_relaxed) + count, std::memory_order_relaxed);
}

void raiseTo(std::atomic<std::uint64_t>& value, std::uint64_t seen) {
    if (seen > value.load(std::memory_order_relaxed)) value.store(seen, std::memory_order_relaxed);
}

void fold(CounterBlock& into, CounterBlock& from) {
    for (std::size_t index = 0; index < CounterCount; ++index) bump(into.counters[index], from.counters[index].load(std::memory_order_relaxed));
    for (std::size_t index = 0; index < FallbackCount; ++index) bump(into.fallbacks[index], from.fallbacks[index].load(std::memory_order_relaxed));
    raiseTo(into.maxBytes, from.maxBytes.load(std::memory_order_relaxed));
    raiseTo(into.maxFollower, from.maxFollower.load(std::memory_order_relaxed));
    raiseTo(into.highWater, from.highWater.load(std::memory_order_relaxed));
}

struct CounterHolder {
    CounterBlock* block = nullptr;

    ~CounterHolder() {
        if (block == nullptr) return;
        auto& registry = counterRegistry();
        std::lock_guard lock(registry.mutex);
        fold(registry.retired, *block);
        std::erase(registry.live, block);
        delete block;
    }
};

struct CounterStorage {};

CounterBlock* ownCounters() noexcept {
    try {
        auto& holder = HostThreadLocal<CounterHolder, CounterStorage>();
        if (holder.block != nullptr) return holder.block;
        auto block = std::make_unique<CounterBlock>();
        auto& registry = counterRegistry();
        std::lock_guard lock(registry.mutex);
        registry.live.push_back(block.get());
        holder.block = block.release();
        return holder.block;
    } catch (...) {
        return nullptr;
    }
}

void noteHighWater(std::size_t used) noexcept {
    if (auto* const block = ownCounters(); block != nullptr) raiseTo(block->highWater, used);
}

std::uint64_t verifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_DRAW_TICKETS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

enum class VerifyKind : std::uint8_t { Flatten, Follower, Thaw, Count };
constexpr std::array<const char*, static_cast<std::size_t>(VerifyKind::Count)> VerifyKindNames{"flatten", "follower", "rebuilt hit"};

bool verifyDue(VerifyKind kind) {
    const auto every = verifyEvery();
    if (every == 0) return false;
    struct OrdinalStorage {};
    auto& ordinals = HostThreadLocal<std::array<std::uint64_t, static_cast<std::size_t>(VerifyKind::Count)>, OrdinalStorage>();
    return ++ordinals[static_cast<std::size_t>(kind)] % every == 0;
}

std::atomic<std::uint64_t> printedDifferences{0};

struct Differences {
    VerifyKind kind;
    bool found = false;

    void Note(const char* field, const std::string& detail) {
        found = true;
        if (printedDifferences.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[draw-tickets-verify] %s %s differs: %s\n", VerifyKindNames[static_cast<std::size_t>(kind)], field, detail.c_str());
    }
    void Check(bool same, const char* field, std::uint64_t mine, std::uint64_t today) {
        if (!same) Note(field, std::to_string(mine) + " against " + std::to_string(today));
    }
    void Count() {
        DrawTickets::Add(DrawTickets::Counter::VerifyChecked);
        if (found) DrawTickets::Add(DrawTickets::Counter::VerifyDiffering);
    }
};

std::uint64_t bits(const void* pointer) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

template <typename T>
T* rawAt(void* base, std::size_t offset) {
    return reinterpret_cast<T*>(static_cast<std::byte*>(base) + offset);
}

template <typename T>
std::span<const T> arrayAt(const DrawTicket& ticket, std::uint32_t offset, std::size_t count) {
    if (count == 0) return {};
    return {ticket.At<T>(offset), count};
}

bool sameMoved(const MovedBuffer& a, const MovedBuffer& b) {
    return a.allocation == b.allocation && a.address == b.address && a.size == b.size && a.words.data() == b.words.data() && a.words.size() == b.words.size() && a.inPlace == b.inPlace && a.importOffset == b.importOffset && a.begin == b.begin && a.bytes == b.bytes && a.behind == b.behind && a.written == b.written;
}

bool sameSelected(const Selected& a, const Selected& b) {
    return a.allocation == b.allocation && a.begin == b.begin && a.bytes == b.bytes && a.offset == b.offset && a.moved == b.moved && a.inPlace == b.inPlace && a.handle == b.handle && a.importOffset == b.importOffset;
}

bool sameRange(const InputRange& a, const InputRange& b) {
    return a.begin == b.begin && a.end == b.end && a.alignment == b.alignment && a.import == b.import && a.handle == b.handle && a.offset == b.offset;
}

bool sameParameters(const Pm4::DrawParameters& a, const Pm4::DrawParameters& b) {
    if (a.indexAddress != b.indexAddress || a.indexCount != b.indexCount || a.indexSize != b.indexSize || a.instanceCount != b.instanceCount || a.flags != b.flags || a.indexed != b.indexed || a.firstVertex != b.firstVertex || a.firstInstance != b.firstInstance || a.indirect.has_value() != b.indirect.has_value()) return false;
    if (!a.indirect.has_value()) return true;
    const auto& x = *a.indirect;
    const auto& y = *b.indirect;
    return x.arguments == y.arguments && x.opcode == y.opcode && x.recordBytes == y.recordBytes && x.stride == y.stride && x.count == y.count && x.countIndirect == y.countIndirect && x.countAddress == y.countAddress && x.indxOffset == y.indxOffset && x.vertexRule == y.vertexRule && x.instanceRule == y.instanceRule;
}

bool sameLayout(const Graphics::VertexInputLayout& a, const Graphics::VertexInputLayout& b) {
    const auto sameBinding = [](const VkVertexInputBindingDescription& x, const VkVertexInputBindingDescription& y) { return x.binding == y.binding && x.stride == y.stride && x.inputRate == y.inputRate; };
    const auto sameAttribute = [](const VkVertexInputAttributeDescription& x, const VkVertexInputAttributeDescription& y) { return x.location == y.location && x.binding == y.binding && x.format == y.format && x.offset == y.offset; };
    return std::ranges::equal(a.bindings, b.bindings, sameBinding) && std::ranges::equal(a.attributes, b.attributes, sameAttribute);
}

template <typename T, typename Same>
void compareArrays(Differences& differences, const char* field, std::span<const T> mine, std::span<const T> today, Same same) {
    if (mine.size() != today.size()) {
        differences.Note(field, "count " + std::to_string(mine.size()) + " against " + std::to_string(today.size()));
        return;
    }
    for (std::size_t index = 0; index < mine.size(); ++index) {
        if (same(mine[index], today[index])) continue;
        differences.Note(field, "entry " + std::to_string(index));
        return;
    }
}

bool hasCapacity(const Side& side) {
    return side.userData.capacity() != 0 || side.flattenedSrt.capacity() != 0 || side.vertexAttributes.capacity() != 0 || side.bindings.capacity() != 0 || side.pushWords.capacity() != 0;
}

bool computeOwners(const Graphics::ShaderResources::DrawBindings& bindings, std::array<KeptOwner, 2>& owners, std::size_t& count) {
    count = 0;
    const auto& snapshots = bindings.snapshots;
    for (std::size_t at = snapshots.size(); at-- > 0;) {
        const auto& buffer = snapshots[at].buffer;
        if (buffer == nullptr) continue;
        const void* const owner = buffer.get();
        if (std::any_of(owners.begin(), owners.begin() + static_cast<std::ptrdiff_t>(count), [&](const KeptOwner& kept) { return kept.owner == owner; })) continue;
        if (count == owners.size()) return false;
        owners[count++] = {owner, buffer->Bytes().size()};
    }
    std::reverse(owners.begin(), owners.begin() + static_cast<std::ptrdiff_t>(count));
    return true;
}

bool ownersFor(const DrawFragment* fragment, std::array<KeptOwner, 2>& owners, std::size_t& count) {
    count = 0;
    if (fragment == nullptr || fragment->bindings == nullptr) return true;
    try {
        return computeOwners(*fragment->bindings, owners, count);
    } catch (const std::exception&) {
        count = 0;
        return false;
    }
}

struct Layout {
    std::size_t ranges = 0;
    std::size_t pairs = 0;
    std::size_t moved = 0;
    std::size_t selected = 0;
    std::size_t stages = 0;
    std::size_t follower = 0;
    std::size_t cold = 0;
    std::size_t reads = 0;
    std::size_t handles = 0;
    std::size_t offsets = 0;
    std::size_t captures = 0;
    std::size_t sides = 0;
    std::size_t bytes = 0;
};

struct Counts {
    std::size_t ranges = 0;
    std::size_t pairs = 0;
    std::size_t moved = 0;
    std::size_t selected = 0;
    std::size_t stages = 0;
    std::size_t reads = 0;
    std::size_t handles = 0;
    std::size_t offsets = 0;
    std::size_t captures = 0;
    std::size_t sides = 0;
};

Layout layoutFor(const Counts& counts) {
    Layout layout;
    std::size_t at = sizeof(DrawTicket);
    const auto take = [&](std::size_t& field, std::size_t alignment, std::size_t bytes) {
        at = alignUp(at, alignment);
        field = at;
        at += bytes;
    };
    take(layout.ranges, alignof(InputRange), counts.ranges * sizeof(InputRange));
    take(layout.pairs, alignof(ReadPair), counts.pairs * sizeof(ReadPair));
    take(layout.moved, alignof(MovedBuffer), counts.moved * sizeof(MovedBuffer));
    take(layout.selected, alignof(Selected), counts.selected * sizeof(Selected));
    take(layout.stages, alignof(Graphics::CompiledShader), counts.stages * sizeof(Graphics::CompiledShader));
    layout.follower = at;
    take(layout.cold, alignof(TicketCold), sizeof(TicketCold));
    take(layout.reads, alignof(DrawPipeline::Range), counts.reads * sizeof(DrawPipeline::Range));
    take(layout.handles, alignof(VkBuffer), counts.handles * sizeof(VkBuffer));
    take(layout.offsets, alignof(VkDeviceSize), counts.offsets * sizeof(VkDeviceSize));
    take(layout.captures, alignof(Capture), counts.captures * sizeof(Capture));
    take(layout.sides, alignof(Side), counts.sides * sizeof(Side));
    layout.bytes = alignUp(at, Line);
    return layout;
}

bool heapInDraft(const DrawRecordHit& draft, const Graphics::CompiledShader& stage, std::size_t& index) {
    const auto at = reinterpret_cast<std::uintptr_t>(stage.heap);
    const auto first = reinterpret_cast<std::uintptr_t>(draft.sides.data());
    const auto end = first + draft.sides.size() * sizeof(Side);
    if (stage.heap == nullptr || at < first || at >= end || (at - first) % sizeof(Side) != 0) return false;
    index = (at - first) / sizeof(Side);
    return true;
}

bool verifyFlatten(const DrawTicket& ticket, const DrawRecordHit& draft, const LeanDrawJob& job, const VulkanDevice* device, const Side* sides, std::span<const KeptOwner> owners) {
    Differences differences{VerifyKind::Flatten};
    const auto& graphics = job.shape->decode->state;
    differences.Check(ticket.commit == draft.record->commit.get(), "commit", bits(ticket.commit), bits(draft.record->commit.get()));
    differences.Check(ticket.device == device, "device", bits(ticket.device), bits(device));
    differences.Check(ticket.state == &graphics, "state", bits(ticket.state), bits(&graphics));
    differences.Check(ticket.queue == job.queue && ticket.opcode == job.opcode, "packet", ticket.opcode, job.opcode);
    differences.Check(ticket.epoch == job.epoch, "epoch", ticket.epoch, job.epoch);
    differences.Check(ticket.shape == job.shapeKey, "shape", ticket.shape, job.shapeKey);
    differences.Check(ticket.trace.token == 0 && ticket.trace.submission == job.serial && ticket.trace.queue == job.queue && ticket.trace.offset == static_cast<std::size_t>(job.packetOffset), "trace", ticket.trace.submission, job.serial);
    if (!sameParameters(ticket.parameters, draft.parameters)) differences.Note("parameters", "a draw parameter");
    const auto stages = ticket.Stages();
    differences.Check(stages.size() == draft.stages.size(), "stage count", stages.size(), draft.stages.size());
    for (std::size_t index = 0; index < std::min(stages.size(), draft.stages.size()); ++index) {
        const auto& mine = stages[index];
        const auto& today = draft.stages[index];
        std::size_t side = 0;
        const auto* expected = heapInDraft(draft, today, side) ? static_cast<const Side*>(sides + side) : today.heap;
        if (mine.stage != today.stage || mine.program != today.program || mine.pushConstantOffset != today.pushConstantOffset || mine.registeredImageViews != today.registeredImageViews || mine.heap != expected) differences.Note("stage", "stage " + std::to_string(index));
    }
    compareArrays(differences, "reads", ticket.Reads(), std::span<const DrawPipeline::Range>(draft.reads), [](const DrawPipeline::Range& a, const DrawPipeline::Range& b) { return a == b; });
    differences.Check(ticket.Has(DrawTicket::Buffers) == draft.preparedBuffers.has_value(), "buffers part", ticket.Has(DrawTicket::Buffers), draft.preparedBuffers.has_value());
    if (draft.preparedBuffers.has_value() && ticket.Has(DrawTicket::Buffers)) {
        const auto& today = *draft.preparedBuffers;
        differences.Check(ticket.blobData == today.blob.data() && ticket.blobBytes == today.blob.size(), "blob", ticket.blobBytes, today.blob.size());
        differences.Check(ticket.generation == today.generation, "generation", ticket.generation, today.generation);
        differences.Check(ticket.depthGuards == today.depthGuards, "depth guards", ticket.depthGuards, today.depthGuards);
        differences.Check(ticket.bufferForget == today.forget && ticket.bufferImports == today.imports && ticket.bufferMapping == today.mapping, "buffer serials", ticket.bufferForget, today.forget);
        differences.Check(ticket.placedDevice == today.placedDevice, "placed device", bits(ticket.placedDevice), bits(today.placedDevice));
        differences.Check(ticket.inPlace == today.inPlace && ticket.aliasesClear == today.aliasesClear && ticket.shortcuts == today.shortcuts && ticket.bufferRefusal == today.refusal, "buffer flags", ticket.shortcuts, today.shortcuts);
        differences.Check(ticket.lookupsDeferred == today.lookupsDeferred && ticket.lookupsForced == today.lookupsForced, "lookups", ticket.lookupsDeferred, today.lookupsDeferred);
        differences.Check(ticket.Has(DrawTicket::PlacedHeld) == (today.placed != nullptr), "placed", ticket.Has(DrawTicket::PlacedHeld), today.placed != nullptr);
        compareArrays(differences, "moved", arrayAt<MovedBuffer>(ticket, ticket.movedAt, ticket.movedCount), std::span<const MovedBuffer>(today.moved), sameMoved);
        compareArrays(differences, "selected", arrayAt<Selected>(ticket, ticket.selectedAt, ticket.selectedCount), std::span<const Selected>(today.selected), sameSelected);
    }
    differences.Check(ticket.Has(DrawTicket::Inputs) == draft.preparedInputs.has_value(), "inputs part", ticket.Has(DrawTicket::Inputs), draft.preparedInputs.has_value());
    if (draft.preparedInputs.has_value() && ticket.Has(DrawTicket::Inputs)) {
        const auto& today = *draft.preparedInputs;
        differences.Check(ticket.inputState == today.state, "inputs state", bits(ticket.inputState), bits(today.state));
        differences.Check(ticket.inputDevice == today.device, "inputs device", bits(ticket.inputDevice), bits(today.device));
        differences.Check(ticket.inputForget == today.forget && ticket.inputImports == today.imports && ticket.inputMapping == today.mapping, "inputs serials", ticket.inputForget, today.forget);
        differences.Check(ticket.indexBytes == today.indexBytes && ticket.maxIndex == today.maxIndex, "index range", ticket.indexBytes, today.indexBytes);
        differences.Check(ticket.indirectRanges == today.indirectRanges && ticket.inputRefusal == today.refusal && ticket.exactCapacity == today.exactCapacity, "inputs flags", ticket.indirectRanges, today.indirectRanges);
        differences.Check(today.draw == &draft.parameters && today.shaders == draft.stages.data() && today.stages == draft.stages.size(), "inputs identity", today.stages, draft.stages.size());
        compareArrays(differences, "input ranges", arrayAt<InputRange>(ticket, ticket.rangesAt, ticket.rangeCount), std::span<const InputRange>(today.ranges), sameRange);
        compareArrays(differences, "in-place reads", arrayAt<ReadPair>(ticket, ticket.pairsAt, ticket.pairCount), std::span<const ReadPair>(today.inPlaceReads), [](const ReadPair& a, const ReadPair& b) { return a == b; });
        compareArrays(differences, "vertex handles", arrayAt<VkBuffer>(ticket, ticket.handlesAt, ticket.handleCount), std::span<const VkBuffer>(today.vertexHandles), [](VkBuffer a, VkBuffer b) { return a == b; });
        compareArrays(differences, "vertex offsets", arrayAt<VkDeviceSize>(ticket, ticket.offsetsAt, ticket.offsetCount), std::span<const VkDeviceSize>(today.vertexOffsets), [](VkDeviceSize a, VkDeviceSize b) { return a == b; });
        if (today.recipe == nullptr || !sameLayout(today.vertexInput, today.recipe->vertexInput)) differences.Note("vertex layout", "the prepared layout is not the recipe's");
    }
    differences.Check(ticket.Has(DrawTicket::Fragment) == draft.fragment.has_value(), "fragment part", ticket.Has(DrawTicket::Fragment), draft.fragment.has_value());
    std::array<KeptOwner, 2> recomputed{};
    std::size_t count = 0;
    const bool held = ownersFor(draft.fragment.has_value() ? &*draft.fragment : nullptr, recomputed, count);
    if (!held || count != owners.size() || !std::equal(owners.begin(), owners.end(), recomputed.begin(), [](const KeptOwner& a, const KeptOwner& b) { return a.owner == b.owner && a.bytes == b.bytes; })) differences.Note("upload chunk owners", "count " + std::to_string(owners.size()) + " against " + std::to_string(count));
    differences.Count();
    return differences.found;
}

}

Graphics::ShaderResources::PreparedBuffersView DrawTicket::BuffersView() const noexcept {
    return {plan.get(), arrayAt<MovedBuffer>(*this, movedAt, movedCount), arrayAt<Selected>(*this, selectedAt, selectedCount), blobBytes == 0 ? std::span<const std::byte>{} : std::span<const std::byte>(blobData, blobBytes), generation, depthGuards, bufferForget, bufferImports, bufferMapping, inPlace, aliasesClear};
}

Graphics::PreparedInputsView DrawTicket::InputsView() const noexcept {
    return {recipe.get(), inputDevice, inputState, &parameters, Stages().data(), stageCount, inputForget, inputImports, inputMapping, indexBytes, maxIndex, indirectRanges, arrayAt<InputRange>(*this, rangesAt, rangeCount), arrayAt<ReadPair>(*this, pairsAt, pairCount)};
}

TicketArena::TicketArena() : storage(new std::byte[RingBytes + Line]) {
    const auto base = reinterpret_cast<std::uintptr_t>(storage.get());
    ring = storage.get() + (alignUp(base, Line) - base);
    blobs.reserve(BlobStack);
    sides.reserve(SideStack);
}

TicketArena::~TicketArena() = default;

TicketArena* TicketArena::Current() noexcept {
    return CurrentArena::Get();
}

TicketArena* TicketArena::ForThread() noexcept {
    if (auto* arena = CurrentArena::Get(); arena != nullptr) return arena;
    if (BoundIndex::Get() == 0) return nullptr;
    try {
        auto* made = new TicketArena;
        CurrentArena::Set(made);
        return made;
    } catch (...) {
        return nullptr;
    }
}

void TicketArena::BindThread(unsigned index) noexcept {
    BoundIndex::Set(index + 1u);
}

void TicketArena::Orphan() noexcept {
    auto* const arena = CurrentArena::Get();
    if (arena == nullptr) return;
    CurrentArena::Set(nullptr);
    if (arena->used == 0) {
        delete arena;
        return;
    }
    arena->orphaned = true;
    auto& registry = orphanRegistry();
    std::lock_guard lock(registry.mutex);
    try {
        registry.orphans.push_back(arena);
    } catch (...) {
        return;
    }
    orphansWaiting.store(static_cast<std::uint32_t>(registry.orphans.size()), std::memory_order_release);
}

void TicketArena::sweepOrphans() noexcept {
    auto& orphans = orphanRegistry().orphans;
    for (auto at = orphans.begin(); at != orphans.end();) {
        auto* const arena = *at;
        while (arena->reclaimTail(true)) {
        }
        if (arena->used != 0) {
            ++at;
            continue;
        }
        delete arena;
        at = orphans.erase(at);
    }
    orphansWaiting.store(static_cast<std::uint32_t>(orphans.size()), std::memory_order_release);
}

void TicketArena::ReclaimOrphans() noexcept {
    auto& registry = orphanRegistry();
    std::lock_guard lock(registry.mutex);
    sweepOrphans();
}

void TicketArena::ReclaimOrphansAtIdle() noexcept {
    auto& registry = orphanRegistry();
    std::unique_lock lock(registry.mutex, std::try_to_lock);
    if (!lock.owns_lock()) return;
    sweepOrphans();
}

bool TicketArena::OrphansWaiting() noexcept {
    return orphansWaiting.load(std::memory_order_relaxed) != 0;
}

bool TicketArena::ReclaimOne() noexcept {
    return reclaimTail(true);
}

void TicketArena::PublishRetired(std::uint64_t seq) noexcept {
    if (seq > retired.load(std::memory_order_relaxed)) retired.store(seq, std::memory_order_release);
}

std::size_t TicketArena::place(std::size_t bytes) noexcept {
    constexpr auto none = std::numeric_limits<std::size_t>::max();
    if (used == 0) {
        head = 0;
        tail = 0;
        wrapEnd = RingBytes;
    }
    std::size_t at = none;
    if (head > tail || used == 0) {
        if (RingBytes - head >= bytes) {
            at = head;
        } else if (tail >= bytes) {
            wrapEnd = head;
            used += RingBytes - head;
            at = 0;
        }
    } else if (head < tail && tail - head >= bytes) {
        at = head;
    }
    if (at == none) return none;
    head = at + bytes;
    used += bytes;
    return at;
}

void* TicketArena::Allocate(std::size_t bytes, const std::shared_ptr<VulkanDevice>& device, TicketFallback& refusal) noexcept {
    constexpr auto none = std::numeric_limits<std::size_t>::max();
    if (used != 0 && heldDevice.get() != device.get()) {
        while (reclaimTail(false)) {
        }
        if (used != 0) {
            refusal = TicketFallback::Device;
            return nullptr;
        }
    }
    auto at = place(bytes);
    if (at == none) {
        while (reclaimTail(false)) {
        }
        at = place(bytes);
        if (at == none) {
            refusal = TicketFallback::Room;
            return nullptr;
        }
    }
    if (heldDevice.get() != device.get()) heldDevice = device;
    noteHighWater(used);
    return ring + at;
}

bool TicketArena::reclaimTail(bool idle) noexcept {
    if (used == 0) return false;
    if (tail == wrapEnd || tail == RingBytes) {
        used -= RingBytes - tail;
        tail = 0;
        wrapEnd = RingBytes;
        if (used == 0) return false;
    }
    auto* const ticket = std::launder(reinterpret_cast<DrawTicket*>(ring + tail));
    if (ticket->seq > retired.load(std::memory_order_acquire) && ticket->abandoned.load(std::memory_order_acquire) == 0) return false;
    const std::size_t bytes = ticket->bytes;
    if (ticket->abandoned.load(std::memory_order_relaxed) == 1) DrawTickets::Add(DrawTickets::Counter::Abandoned);
    destroy(*ticket);
    tail += bytes;
    used -= bytes;
    DrawTickets::Add(idle ? DrawTickets::Counter::ReclaimedIdle : DrawTickets::Counter::ReclaimedInJob);
    if (used == 0) {
        head = 0;
        tail = 0;
        wrapEnd = RingBytes;
        if (idle) heldDevice.reset();
    }
    return true;
}

void TicketArena::destroy(DrawTicket& ticket) noexcept {
    if (ticket.sideCount != 0) {
        auto* const ticketSides = ticket.At<Side>(ticket.sidesAt);
        for (std::size_t index = 0; index < ticket.sideCount; ++index) {
            auto& side = ticketSides[index];
            if (!orphaned && sides.size() < SideStack && hasCapacity(side)) sides.push_back(std::move(side));
            side.~Side();
        }
    }
    if (ticket.captureCount != 0) {
        auto* const captures = ticket.At<Capture>(ticket.capturesAt);
        for (std::size_t index = 0; index < ticket.captureCount; ++index) captures[index].~Capture();
    }
    auto& cold = ticket.Cold();
    if (!orphaned && blobs.size() < BlobStack && cold.blob.capacity() != 0) {
        cold.blob.clear();
        blobs.push_back(std::move(cold.blob));
    }
    cold.~TicketCold();
    ticket.~DrawTicket();
}

void TicketArena::RefillBlob(std::vector<std::byte>& blob) noexcept {
    if (blob.capacity() != 0 || blobs.empty()) return;
    blob = std::move(blobs.back());
    blobs.pop_back();
}

void TicketArena::RefillSide(Side& side) noexcept {
    if (hasCapacity(side) || sides.empty()) return;
    side = std::move(sides.back());
    sides.pop_back();
}

void RefillTicketSide(Side& side) noexcept {
    if (auto* const arena = TicketArena::Current(); arena != nullptr) arena->RefillSide(side);
}

TicketRetire RetireOf(const void* ticket) noexcept {
    if (ticket == nullptr) return {};
    const auto& held = *static_cast<const DrawTicket*>(ticket);
    return {held.arena, held.seq};
}

void AbandonTicket(void* ticket) noexcept {
    if (ticket == nullptr) return;
    static_cast<DrawTicket*>(ticket)->abandoned.store(1, std::memory_order_release);
}

DrawTicket* FlattenTicket(TicketArena& arena, DrawRecordHit& draft, const LeanDrawJob& job, const std::shared_ptr<VulkanDevice>& device, TicketFallback& refusal) {
    auto* const buffers = draft.preparedBuffers.has_value() ? &*draft.preparedBuffers : nullptr;
    auto* const inputs = draft.preparedInputs.has_value() ? &*draft.preparedInputs : nullptr;
    auto* const fragment = draft.fragment.has_value() ? &*draft.fragment : nullptr;
    if (draft.record == nullptr || draft.record->commit == nullptr || draft.owner == nullptr) {
        refusal = TicketFallback::Identity;
        return nullptr;
    }
    if (inputs != nullptr && (inputs->draw != &draft.parameters || inputs->shaders != draft.stages.data() || inputs->stages != draft.stages.size())) {
        refusal = TicketFallback::Identity;
        return nullptr;
    }
    std::array<KeptOwner, 2> owners{};
    std::size_t ownerCount = 0;
    if (!ownersFor(fragment, owners, ownerCount)) {
        refusal = TicketFallback::Owners;
        return nullptr;
    }
    Counts counts;
    counts.stages = draft.stages.size();
    counts.sides = draft.sides.size();
    counts.captures = draft.captures.size();
    counts.reads = draft.reads.size();
    if (buffers != nullptr) {
        counts.moved = buffers->moved.size();
        counts.selected = buffers->selected.size();
    }
    if (inputs != nullptr) {
        counts.ranges = inputs->ranges.size();
        counts.pairs = inputs->inPlaceReads.size();
        counts.handles = inputs->vertexHandles.size();
        counts.offsets = inputs->vertexOffsets.size();
    }
    constexpr std::size_t CountLimit = std::numeric_limits<std::uint16_t>::max();
    const auto largest = std::max({counts.stages, counts.sides, counts.captures, counts.reads, counts.moved, counts.selected, counts.ranges, counts.pairs, counts.handles, counts.offsets});
    if (largest > CountLimit) {
        refusal = TicketFallback::Size;
        return nullptr;
    }
    const auto layout = layoutFor(counts);
    if (layout.bytes > TicketArena::MaxTicketBytes) {
        refusal = TicketFallback::Size;
        return nullptr;
    }
    if (device == nullptr) {
        refusal = TicketFallback::Device;
        return nullptr;
    }
    void* const place = arena.Allocate(layout.bytes, device, refusal);
    if (place == nullptr) return nullptr;
    auto* const ticket = new (place) DrawTicket{};
    ticket->arena = &arena;
    ticket->seq = arena.NextSeq();
    ticket->bytes = static_cast<std::uint32_t>(layout.bytes);
    ticket->followerBytes = static_cast<std::uint32_t>(layout.follower);
    ticket->parts = (buffers != nullptr ? DrawTicket::Buffers : 0u) | (inputs != nullptr ? DrawTicket::Inputs : 0u) | (fragment != nullptr ? DrawTicket::Fragment : 0u) | (buffers != nullptr && buffers->placed != nullptr ? DrawTicket::PlacedHeld : 0u);
    ticket->stageCount = static_cast<std::uint16_t>(counts.stages);
    ticket->movedCount = static_cast<std::uint16_t>(counts.moved);
    ticket->selectedCount = static_cast<std::uint16_t>(counts.selected);
    ticket->rangeCount = static_cast<std::uint16_t>(counts.ranges);
    ticket->pairCount = static_cast<std::uint16_t>(counts.pairs);
    ticket->handleCount = static_cast<std::uint16_t>(counts.handles);
    ticket->offsetCount = static_cast<std::uint16_t>(counts.offsets);
    ticket->readCount = static_cast<std::uint16_t>(counts.reads);
    ticket->ownerCount = static_cast<std::uint16_t>(ownerCount);
    ticket->rangesAt = static_cast<std::uint32_t>(layout.ranges);
    ticket->pairsAt = static_cast<std::uint32_t>(layout.pairs);
    ticket->movedAt = static_cast<std::uint32_t>(layout.moved);
    ticket->selectedAt = static_cast<std::uint32_t>(layout.selected);
    ticket->stagesAt = static_cast<std::uint32_t>(layout.stages);
    ticket->coldAt = static_cast<std::uint32_t>(layout.cold);
    ticket->readsAt = static_cast<std::uint32_t>(layout.reads);
    ticket->handlesAt = static_cast<std::uint32_t>(layout.handles);
    ticket->offsetsAt = static_cast<std::uint32_t>(layout.offsets);
    ticket->capturesAt = static_cast<std::uint32_t>(layout.captures);
    ticket->sidesAt = static_cast<std::uint32_t>(layout.sides);
    ticket->commit = draft.record->commit.get();
    ticket->device = device.get();
    const auto& graphics = job.shape->decode->state;
    ticket->state = &graphics;
    ticket->queue = job.queue;
    ticket->opcode = job.opcode;
    ticket->epoch = job.epoch;
    ticket->shape = job.shapeKey;
    ticket->trace = {0, job.serial, job.queue, static_cast<std::size_t>(job.packetOffset), graphics.hasColorTarget ? graphics.color.address : 0};
    ticket->parameters = draft.parameters;
    if (buffers != nullptr) {
        ticket->blobData = buffers->blob.data();
        ticket->blobBytes = buffers->blob.size();
        ticket->generation = buffers->generation;
        ticket->depthGuards = buffers->depthGuards;
        ticket->bufferForget = buffers->forget;
        ticket->bufferImports = buffers->imports;
        ticket->bufferMapping = buffers->mapping;
        ticket->placedDevice = buffers->placedDevice;
        ticket->lookupsDeferred = buffers->lookupsDeferred;
        ticket->lookupsForced = buffers->lookupsForced;
        ticket->bufferRefusal = buffers->refusal;
        ticket->shortcuts = buffers->shortcuts;
        ticket->inPlace = buffers->inPlace;
        ticket->aliasesClear = buffers->aliasesClear;
        std::uninitialized_copy_n(buffers->moved.data(), counts.moved, rawAt<MovedBuffer>(ticket, layout.moved));
        std::uninitialized_copy_n(buffers->selected.data(), counts.selected, rawAt<Selected>(ticket, layout.selected));
    }
    if (inputs != nullptr) {
        ticket->inputState = inputs->state;
        ticket->inputDevice = inputs->device;
        ticket->inputForget = inputs->forget;
        ticket->inputImports = inputs->imports;
        ticket->inputMapping = inputs->mapping;
        ticket->indexBytes = inputs->indexBytes;
        ticket->indirectRanges = inputs->indirectRanges;
        ticket->maxIndex = inputs->maxIndex;
        ticket->inputRefusal = inputs->refusal;
        ticket->exactCapacity = inputs->exactCapacity;
        std::uninitialized_copy_n(inputs->ranges.data(), counts.ranges, rawAt<InputRange>(ticket, layout.ranges));
        std::uninitialized_copy_n(inputs->inPlaceReads.data(), counts.pairs, rawAt<ReadPair>(ticket, layout.pairs));
        std::uninitialized_copy_n(inputs->vertexHandles.data(), counts.handles, rawAt<VkBuffer>(ticket, layout.handles));
        std::uninitialized_copy_n(inputs->vertexOffsets.data(), counts.offsets, rawAt<VkDeviceSize>(ticket, layout.offsets));
    }
    std::copy_n(owners.begin(), ownerCount, ticket->owners.begin());
    auto* const sides = rawAt<Side>(ticket, layout.sides);
    auto* const stages = rawAt<Graphics::CompiledShader>(ticket, layout.stages);
    std::uninitialized_copy_n(draft.stages.data(), counts.stages, stages);
    for (std::size_t index = 0; index < counts.stages; ++index) {
        std::size_t side = 0;
        if (heapInDraft(draft, draft.stages[index], side)) stages[index].heap = sides + side;
    }
    std::uninitialized_copy_n(draft.reads.data(), counts.reads, rawAt<DrawPipeline::Range>(ticket, layout.reads));
    new (rawAt<TicketCold>(ticket, layout.cold)) TicketCold{};
    if (verifyDue(VerifyKind::Flatten)) {
        bool differs = false;
        try {
            differs = verifyFlatten(*ticket, draft, job, device.get(), sides, ticket->Owners());
        } catch (const std::exception&) {
            differs = false;
        }
        if (differs) {
            ticket->abandoned.store(2, std::memory_order_release);
            refusal = TicketFallback::Verify;
            return nullptr;
        }
    }
    auto& cold = ticket->Cold();
    if (buffers != nullptr) {
        ticket->plan = std::move(buffers->plan);
        cold.blob = std::move(buffers->blob);
        cold.placed = std::move(buffers->placed);
    }
    if (inputs != nullptr) ticket->recipe = std::move(inputs->recipe);
    if (fragment != nullptr) ticket->fragment = std::move(*fragment);
    auto* const captures = rawAt<Capture>(ticket, layout.captures);
    for (std::size_t index = 0; index < counts.captures; ++index) new (captures + index) Capture(std::move(draft.captures[index]));
    ticket->captureCount = static_cast<std::uint16_t>(counts.captures);
    for (std::size_t index = 0; index < counts.sides; ++index) new (sides + index) Side(std::move(draft.sides[index]));
    ticket->sideCount = static_cast<std::uint16_t>(counts.sides);
    cold.owner = std::move(draft.owner);
    cold.decode = job.shape->decode;
    if (auto* const block = ownCounters(); block != nullptr) {
        bump(block->counters[static_cast<std::size_t>(DrawTickets::Counter::Made)], 1);
        bump(block->counters[static_cast<std::size_t>(DrawTickets::Counter::Bytes)], layout.bytes);
        bump(block->counters[static_cast<std::size_t>(DrawTickets::Counter::FollowerBytes)], layout.follower);
        bump(block->counters[static_cast<std::size_t>(DrawTickets::Counter::Owners)], ownerCount);
        raiseTo(block->maxBytes, layout.bytes);
        raiseTo(block->maxFollower, layout.follower);
    }
    return ticket;
}

std::span<const KeptOwner> FollowerOwners(const DrawTicket& ticket, std::array<KeptOwner, 2>& recomputed) {
    if (!verifyDue(VerifyKind::Follower)) return ticket.Owners();
    Differences differences{VerifyKind::Follower};
    std::size_t count = 0;
    const bool held = ownersFor(ticket.Has(DrawTicket::Fragment) ? &ticket.fragment : nullptr, recomputed, count);
    const auto kept = ticket.Owners();
    if (!held || count != kept.size() || !std::equal(kept.begin(), kept.end(), recomputed.begin(), [](const KeptOwner& a, const KeptOwner& b) { return a.owner == b.owner && a.bytes == b.bytes; })) differences.Note("upload chunk owners", "count " + std::to_string(kept.size()) + " against " + std::to_string(count));
    differences.Count();
    if (differences.found) return {};
    return kept;
}

namespace {

struct ThawScratch {
    DrawRecordHit hit;
    bool busy = false;
};

struct ThawStorage {};

bool thawTimed() {
    struct OrdinalStorage {};
    auto& ordinal = HostThreadLocal<std::uint64_t, OrdinalStorage>();
    return (ordinal++ & 7u) == 0;
}

void releaseThaw(DrawRecordHit& out) noexcept {
    if (out.preparedBuffers.has_value()) {
        out.spareBuffers = std::move(*out.preparedBuffers);
        out.preparedBuffers.reset();
    }
    if (out.preparedInputs.has_value()) {
        out.spareInputs = std::move(*out.preparedInputs);
        out.preparedInputs.reset();
    }
    out.fragment.reset();
    out.record.reset();
    out.owner.reset();
    out.device.reset();
}

void verifyThaw(const DrawTicket& ticket, const DrawRecordHit& out, const Graphics::ShaderResources::PreparedBuffersView& buffers, const Graphics::PreparedInputsView& inputs) {
    Differences differences{VerifyKind::Thaw};
    if (!sameParameters(out.parameters, ticket.parameters)) differences.Note("parameters", "a draw parameter");
    compareArrays(differences, "stages", std::span<const Graphics::CompiledShader>(out.stages), ticket.Stages(), [](const Graphics::CompiledShader& a, const Graphics::CompiledShader& b) { return a.stage == b.stage && a.program == b.program && a.pushConstantOffset == b.pushConstantOffset && a.registeredImageViews == b.registeredImageViews && a.heap == b.heap; });
    differences.Check(out.direct && out.record != nullptr && out.record->commit.get() == ticket.commit, "record", bits(out.record != nullptr ? out.record->commit.get() : nullptr), bits(ticket.commit));
    differences.Check(out.device.get() == ticket.device, "device", bits(out.device.get()), bits(ticket.device));
    differences.Check(out.preparedBuffers.has_value() == ticket.Has(DrawTicket::Buffers), "buffers part", out.preparedBuffers.has_value(), ticket.Has(DrawTicket::Buffers));
    if (out.preparedBuffers.has_value() && ticket.Has(DrawTicket::Buffers)) {
        const auto mine = out.preparedBuffers->View();
        differences.Check(mine.plan == buffers.plan, "plan", bits(mine.plan), bits(buffers.plan));
        differences.Check(mine.blob.data() == buffers.blob.data() && mine.blob.size() == buffers.blob.size(), "blob", mine.blob.size(), buffers.blob.size());
        differences.Check(mine.generation == buffers.generation && mine.depthGuards == buffers.depthGuards && mine.forget == buffers.forget && mine.imports == buffers.imports && mine.mapping == buffers.mapping, "buffer serials", mine.generation, buffers.generation);
        differences.Check(mine.inPlace == buffers.inPlace && mine.aliasesClear == buffers.aliasesClear, "buffer flags", mine.inPlace, buffers.inPlace);
        differences.Check((out.preparedBuffers->placed != nullptr) == ticket.Has(DrawTicket::PlacedHeld), "placed", out.preparedBuffers->placed != nullptr, ticket.Has(DrawTicket::PlacedHeld));
        compareArrays(differences, "moved", mine.moved, buffers.moved, sameMoved);
        compareArrays(differences, "selected", mine.selected, buffers.selected, sameSelected);
    }
    differences.Check(out.preparedInputs.has_value() == ticket.Has(DrawTicket::Inputs), "inputs part", out.preparedInputs.has_value(), ticket.Has(DrawTicket::Inputs));
    if (out.preparedInputs.has_value() && ticket.Has(DrawTicket::Inputs)) {
        const auto& prepared = *out.preparedInputs;
        const auto mine = prepared.View();
        differences.Check(mine.recipe == inputs.recipe && mine.device == inputs.device && mine.state == inputs.state, "inputs objects", bits(mine.recipe), bits(inputs.recipe));
        differences.Check(mine.draw == &out.parameters && mine.shaders == out.stages.data() && mine.stages == out.stages.size(), "inputs identity", mine.stages, out.stages.size());
        differences.Check(mine.forget == inputs.forget && mine.imports == inputs.imports && mine.mapping == inputs.mapping, "inputs serials", mine.forget, inputs.forget);
        differences.Check(mine.indexBytes == inputs.indexBytes && mine.maxIndex == inputs.maxIndex && mine.indirectRanges == inputs.indirectRanges, "index range", mine.indexBytes, inputs.indexBytes);
        compareArrays(differences, "input ranges", mine.ranges, inputs.ranges, sameRange);
        compareArrays(differences, "in-place reads", mine.inPlaceReads, inputs.inPlaceReads, [](const ReadPair& a, const ReadPair& b) { return a == b; });
        compareArrays(differences, "vertex handles", std::span<const VkBuffer>(prepared.vertexHandles), arrayAt<VkBuffer>(ticket, ticket.handlesAt, ticket.handleCount), [](VkBuffer a, VkBuffer b) { return a == b; });
        compareArrays(differences, "vertex offsets", std::span<const VkDeviceSize>(prepared.vertexOffsets), arrayAt<VkDeviceSize>(ticket, ticket.offsetsAt, ticket.offsetCount), [](VkDeviceSize a, VkDeviceSize b) { return a == b; });
        if (prepared.recipe == nullptr || !sameLayout(prepared.vertexInput, prepared.recipe->vertexInput)) differences.Note("vertex layout", "the rebuilt layout is not the recipe's");
    }
    differences.Check(out.fragment.has_value() == ticket.Has(DrawTicket::Fragment), "fragment part", out.fragment.has_value(), ticket.Has(DrawTicket::Fragment));
    differences.Count();
}

}

ThawedTicket::ThawedTicket(DrawTicket& held) : ticket(held) {
    timed = thawTimed();
    const auto began = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool counted = AgcDriver::HostAllocationCountsOn();
    std::uint64_t allocationsBefore = 0;
    std::uint64_t freesBefore = 0;
    if (counted) AgcDriver::HostAllocationCounts(allocationsBefore, freesBefore);
    const bool verify = verifyDue(VerifyKind::Thaw);
    auto& scratch = HostThreadLocal<ThawScratch, ThawStorage>();
    if (!scratch.busy) {
        scratch.busy = true;
        busy = &scratch.busy;
        hit = &scratch.hit;
    } else {
        nested = std::make_unique<DrawRecordHit>();
        hit = nested.get();
    }
    auto& out = *hit;
    auto& cold = ticket.Cold();
    const auto buffersBefore = verify ? ticket.BuffersView() : Graphics::ShaderResources::PreparedBuffersView{};
    const auto inputsBefore = verify ? ticket.InputsView() : Graphics::PreparedInputsView{};
    try {
        out.record = std::shared_ptr<const DrawRecord>(std::shared_ptr<const DrawRecord>{}, static_cast<const DirectClass*>(cold.owner.get())->record.get());
        out.owner = cold.owner;
        out.device = ticket.arena->HeldDevice();
        out.direct = true;
        out.planned = false;
        out.parameters = ticket.parameters;
        const auto stages = ticket.Stages();
        out.stages.assign(stages.begin(), stages.end());
        if (ticket.Has(DrawTicket::Buffers)) {
            auto& prepared = out.preparedBuffers.emplace(std::move(out.spareBuffers));
            const auto moved = arrayAt<MovedBuffer>(ticket, ticket.movedAt, ticket.movedCount);
            const auto selected = arrayAt<Selected>(ticket, ticket.selectedAt, ticket.selectedCount);
            prepared.moved.assign(moved.begin(), moved.end());
            prepared.selected.assign(selected.begin(), selected.end());
            prepared.refusal = ticket.bufferRefusal;
            prepared.generation = ticket.generation;
            prepared.depthGuards = ticket.depthGuards;
            prepared.forget = ticket.bufferForget;
            prepared.imports = ticket.bufferImports;
            prepared.mapping = ticket.bufferMapping;
            prepared.inPlace = ticket.inPlace;
            prepared.placedDevice = ticket.placedDevice;
            prepared.aliasesClear = ticket.aliasesClear;
            prepared.shortcuts = ticket.shortcuts;
            prepared.lookupsDeferred = ticket.lookupsDeferred;
            prepared.lookupsForced = ticket.lookupsForced;
        }
        if (ticket.Has(DrawTicket::Inputs)) {
            auto& prepared = out.preparedInputs.emplace(std::move(out.spareInputs));
            const auto ranges = arrayAt<InputRange>(ticket, ticket.rangesAt, ticket.rangeCount);
            const auto pairs = arrayAt<ReadPair>(ticket, ticket.pairsAt, ticket.pairCount);
            const auto handles = arrayAt<VkBuffer>(ticket, ticket.handlesAt, ticket.handleCount);
            const auto offsets = arrayAt<VkDeviceSize>(ticket, ticket.offsetsAt, ticket.offsetCount);
            prepared.refusal = ticket.inputRefusal;
            prepared.state = ticket.inputState;
            prepared.draw = &out.parameters;
            prepared.shaders = out.stages.data();
            prepared.stages = out.stages.size();
            prepared.device = ticket.inputDevice;
            prepared.forget = ticket.inputForget;
            prepared.imports = ticket.inputImports;
            prepared.mapping = ticket.inputMapping;
            prepared.indexBytes = ticket.indexBytes;
            prepared.maxIndex = ticket.maxIndex;
            if (ticket.recipe != nullptr) {
                prepared.vertexInput = ticket.recipe->vertexInput;
            } else {
                prepared.vertexInput.bindings.clear();
                prepared.vertexInput.attributes.clear();
            }
            prepared.vertexHandles.assign(handles.begin(), handles.end());
            prepared.vertexOffsets.assign(offsets.begin(), offsets.end());
            prepared.inPlaceReads.assign(pairs.begin(), pairs.end());
            prepared.ranges.assign(ranges.begin(), ranges.end());
            prepared.indirectRanges = ticket.indirectRanges;
            prepared.exactCapacity = ticket.exactCapacity;
        }
    } catch (...) {
        releaseThaw(out);
        if (busy != nullptr) *busy = false;
        throw;
    }
    if (ticket.Has(DrawTicket::Buffers)) {
        auto& prepared = *out.preparedBuffers;
        prepared.plan = std::move(ticket.plan);
        prepared.blob.swap(cold.blob);
        prepared.placed = std::move(cold.placed);
    }
    if (ticket.Has(DrawTicket::Inputs)) out.preparedInputs->recipe = std::move(ticket.recipe);
    if (ticket.Has(DrawTicket::Fragment)) out.fragment.emplace(std::move(ticket.fragment));
    if (counted) {
        std::uint64_t allocations = 0;
        std::uint64_t frees = 0;
        AgcDriver::HostAllocationCounts(allocations, frees);
        DrawTickets::Add(DrawTickets::Counter::ThawAllocations, allocations - allocationsBefore);
        DrawTickets::Add(DrawTickets::Counter::ThawsCounted);
    }
    if (verify) {
        try {
            verifyThaw(ticket, out, buffersBefore, inputsBefore);
        } catch (const std::exception&) {
        }
    }
    DrawTickets::Add(DrawTickets::Counter::Thaws);
    if (timed) DrawTickets::Add(DrawTickets::Counter::ThawNanoseconds, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - began).count()));
}

ThawedTicket::~ThawedTicket() {
    const auto began = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto& out = *hit;
    auto& cold = ticket.Cold();
    if (out.preparedBuffers.has_value() && ticket.Has(DrawTicket::Buffers)) {
        auto& prepared = *out.preparedBuffers;
        ticket.plan = std::move(prepared.plan);
        cold.placed = std::move(prepared.placed);
        prepared.blob.swap(cold.blob);
    }
    if (out.preparedInputs.has_value() && ticket.Has(DrawTicket::Inputs)) ticket.recipe = std::move(out.preparedInputs->recipe);
    if (out.fragment.has_value() && ticket.Has(DrawTicket::Fragment)) ticket.fragment = std::move(*out.fragment);
    releaseThaw(out);
    if (busy != nullptr) *busy = false;
    if (!timed) return;
    DrawTickets::Add(DrawTickets::Counter::ThawNanoseconds, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - began).count()));
    DrawTickets::Add(DrawTickets::Counter::ThawsTimed);
}

namespace DrawTickets {

void Add(Counter counter, std::uint64_t count) {
    if (auto* const block = ownCounters(); block != nullptr) bump(block->counters[static_cast<std::size_t>(counter)], count);
}

void AddFallback(TicketFallback reason) {
    const auto index = static_cast<std::size_t>(reason);
    if (index >= FallbackCount) return;
    if (auto* const block = ownCounters(); block != nullptr) bump(block->fallbacks[index], 1);
}

void NoteCommit(bool followed, bool prefetched) {
    struct Tally {
        std::uint64_t commits = 0;
        std::uint64_t follows = 0;
        std::uint64_t prefetches = 0;
    };
    struct TallyStorage {};
    auto& tally = HostThreadLocal<Tally, TallyStorage>();
    ++tally.commits;
    if (followed) ++tally.follows;
    if (prefetched) ++tally.prefetches;
    if (tally.commits < 64) return;
    if (auto* const block = ownCounters(); block != nullptr) {
        bump(block->counters[static_cast<std::size_t>(Counter::Committed)], tally.commits);
        bump(block->counters[static_cast<std::size_t>(Counter::Followed)], tally.follows);
        bump(block->counters[static_cast<std::size_t>(Counter::PrefetchedFollowers)], tally.prefetches);
    }
    tally = {};
}

std::uint64_t Total(Counter counter) {
    const auto index = static_cast<std::size_t>(counter);
    if (index >= CounterCount) return 0;
    auto& registry = counterRegistry();
    std::lock_guard lock(registry.mutex);
    auto total = registry.retired.counters[index].load(std::memory_order_relaxed);
    for (const auto* block : registry.live) total += block->counters[index].load(std::memory_order_relaxed);
    return total;
}

void Report() {
    struct Totals {
        std::array<std::uint64_t, CounterCount> counters{};
        std::array<std::uint64_t, FallbackCount> fallbacks{};
        std::uint64_t maxBytes = 0;
        std::uint64_t maxFollower = 0;
        std::uint64_t highWater = 0;
    };
    static Totals reported;
    static std::uint64_t prefetchesReported = 0;
    Totals now;
    {
        auto& registry = counterRegistry();
        std::lock_guard lock(registry.mutex);
        const auto add = [&](CounterBlock& block) {
            for (std::size_t index = 0; index < CounterCount; ++index) now.counters[index] += block.counters[index].load(std::memory_order_relaxed);
            for (std::size_t index = 0; index < FallbackCount; ++index) now.fallbacks[index] += block.fallbacks[index].load(std::memory_order_relaxed);
            now.maxBytes = std::max<std::uint64_t>(now.maxBytes, block.maxBytes.exchange(0, std::memory_order_relaxed));
            now.maxFollower = std::max<std::uint64_t>(now.maxFollower, block.maxFollower.exchange(0, std::memory_order_relaxed));
            now.highWater = std::max<std::uint64_t>(now.highWater, block.highWater.exchange(0, std::memory_order_relaxed));
        };
        add(registry.retired);
        for (auto* block : registry.live) add(*block);
    }
    std::array<std::uint64_t, CounterCount> since{};
    std::array<std::uint64_t, FallbackCount> fallbacks{};
    for (std::size_t index = 0; index < CounterCount; ++index) since[index] = now.counters[index] - reported.counters[index];
    for (std::size_t index = 0; index < FallbackCount; ++index) fallbacks[index] = now.fallbacks[index] - reported.fallbacks[index];
    reported.counters = now.counters;
    reported.fallbacks = now.fallbacks;
    const auto prefetches = DrawPipeline::TicketPrefetches();
    const auto prefetched = prefetches - prefetchesReported;
    prefetchesReported = prefetches;
    const auto at = [&](Counter counter) { return since[static_cast<std::size_t>(counter)]; };
    std::uint64_t fallen = 0;
    std::string reasons;
    for (std::size_t index = 0; index < FallbackCount; ++index) {
        fallen += fallbacks[index];
        if (fallbacks[index] == 0) continue;
        reasons += reasons.empty() ? " (" : ", ";
        reasons += FallbackNames[index];
        reasons += ' ';
        reasons += std::to_string(fallbacks[index]);
    }
    if (!reasons.empty()) reasons += ')';
    const auto eligible = at(Counter::Eligible);
    const auto made = at(Counter::Made);
    const auto committed = at(Counter::Committed);
    const auto released = at(Counter::ReclaimedIdle) + at(Counter::ReclaimedInJob);
    if (verifyEvery() != 0 && at(Counter::VerifyChecked) != 0) AgcDriver::ProfilePrint_nid_no_patch("[draw-tickets-verify] checked %llu differing %llu\n", static_cast<unsigned long long>(at(Counter::VerifyChecked)), static_cast<unsigned long long>(at(Counter::VerifyDiffering)));
    if (eligible == 0 && made == 0 && committed == 0 && released == 0 && fallen == 0) return;
    const auto share = [](std::uint64_t part, std::uint64_t whole) { return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole); };
    const auto mean = [](std::uint64_t total, std::uint64_t count) { return count == 0 ? 0.0 : static_cast<double>(total) / static_cast<double>(count); };
    AgcDriver::ProfilePrint_nid_no_patch("[draw-tickets] %llu tickets of %llu direct hits at the commit (%.1f%%), today's format %llu (%.1f%%)%s; bytes per ticket %.0f (max %llu), follower part %.0f (max %llu); ring high water %.1f KiB; released %llu (%.1f%% at idle points), abandoned %llu; ticket commits %llu: followers %llu (%llu with a prefetched ticket), rebuilt hits %llu (%.1f%% of ticket commits, %.0f ns each over %llu timed, %.1f allocations each over %llu counted); upload chunk owners per ticket %.2f; prefetches issued %llu\n",
        static_cast<unsigned long long>(made), static_cast<unsigned long long>(eligible), share(made, eligible), static_cast<unsigned long long>(fallen), share(fallen, eligible), reasons.c_str(),
        mean(at(Counter::Bytes), made), static_cast<unsigned long long>(now.maxBytes), mean(at(Counter::FollowerBytes), made), static_cast<unsigned long long>(now.maxFollower),
        static_cast<double>(now.highWater) / 1024.0,
        static_cast<unsigned long long>(released), share(at(Counter::ReclaimedIdle), released), static_cast<unsigned long long>(at(Counter::Abandoned)),
        static_cast<unsigned long long>(committed), static_cast<unsigned long long>(at(Counter::Followed)), static_cast<unsigned long long>(at(Counter::PrefetchedFollowers)),
        static_cast<unsigned long long>(at(Counter::Thaws)), share(at(Counter::Thaws), committed), mean(at(Counter::ThawNanoseconds), at(Counter::ThawsTimed)), static_cast<unsigned long long>(at(Counter::ThawsTimed)), mean(at(Counter::ThawAllocations), at(Counter::ThawsCounted)), static_cast<unsigned long long>(at(Counter::ThawsCounted)),
        mean(at(Counter::Owners), made), static_cast<unsigned long long>(prefetched));
}

}

}
