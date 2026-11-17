#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DISPATCHCACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DISPATCHCACHE_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/Recipe.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "IntermediateRepresentation/IrMetadata/ResourcePlan.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <span>
#include <utility>
#include <vector>
#include "prx/libc/include/general/AtomicSharedPtr.hpp"

namespace ShaderRecompiler {
struct SideForwards;
}

namespace AgcDriver::DriverDetail {

inline bool CaptureRegisteredViewsCurrent(std::span<const ShaderRecompiler::RegisteredImageView> views) {
    if (views.empty()) return true;
    constexpr std::size_t Few = 8;
    std::array<std::uintptr_t, Few> fewAddresses;
    std::array<std::shared_ptr<const GuestAllocations::Range>, Few> fewRanges;
    std::vector<std::uintptr_t> manyAddresses;
    std::vector<std::shared_ptr<const GuestAllocations::Range>> manyRanges;
    auto* addresses = fewAddresses.data();
    auto* ranges = fewRanges.data();
    if (views.size() > Few) {
        manyAddresses.resize(views.size());
        manyRanges.resize(views.size());
        addresses = manyAddresses.data();
        ranges = manyRanges.data();
    }
    for (std::size_t index = 0; index < views.size(); ++index) addresses[index] = static_cast<std::uintptr_t>(views[index].address);
    GuestAllocations::GuestAllocationsLookupEach_nid_postfix(addresses, views.size(), ranges);
    for (std::size_t index = 0; index < views.size(); ++index) {
        const auto& view = views[index];
        const auto& range = ranges[index];
        if (view.hasRange != (range != nullptr)) return false;
        if (range != nullptr && (range->address != view.rangeAddress || range->bytes != view.rangeBytes || range->allocationAddress != view.allocationAddress || range->allocationBytes != view.allocationBytes || range->backingSerial != view.backingSerial || range->readable != view.readable || range->writable != view.writable || range->gpu != view.gpu)) return false;
    }
    return true;
}

inline constexpr std::size_t MaxDispatchVariants = 8;

struct DispatchAddressPosition {
    std::uint32_t position = 0;
    std::uint32_t binding = 0;
    std::uint32_t word = 0;
    std::uint32_t mask = 0;
    bool guarded = false;
};

struct DispatchVariant {

    std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
    std::vector<std::uint32_t> words;
    std::vector<SharedWords> shared;
    std::uint64_t forgetSerial = 0;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
    std::shared_ptr<const std::vector<ShaderRecompiler::RegisteredImageView>> registeredViews;
    std::array<std::uint32_t, 4> meshIndexDescriptor{};

    std::shared_ptr<const ShaderSnapshot> shader;
    std::shared_ptr<const PreparedShaderState::UserDataLink> linking;

    AtomicSharedPtr<const Recipe> recipe;

    static constexpr std::uint32_t NoFlatBinding = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> dataPositions;
    std::vector<std::uint32_t> dataSlots;
    std::uint32_t flatBinding = NoFlatBinding;

    std::uint32_t pushOffset = 0;
    std::vector<std::int8_t> runBases;
    std::uint64_t relocatableWords = 0;
    std::uint64_t userWordsWalked = 0;
    std::uint64_t valueWords = 0;
    std::vector<std::array<std::uint32_t, 3>> valueSlots;
    std::vector<std::array<std::uint32_t, 3>> descriptorPatches;
    std::uint64_t calleeWords = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> calleeReads;
    std::shared_ptr<DispatchVariant> donor;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> shaderDataPatches;
    std::uint64_t userDataMask = ~std::uint64_t{0};
    std::shared_ptr<const ShaderRecompiler::ShaderVertexStageInfo> vertexInfo;
    std::shared_ptr<const ShaderRecompiler::HeapStageWords> ringTemplate;
    // The words a relocation patched (descriptor and shader data words, forwarded push constant
    // dwords) beside `compiled`, which stays the donor's: read through DrawGuestWords and
    // AssemblePushConstants (Driver::applyShaderDataPatches).
    std::shared_ptr<const ShaderRecompiler::HeapStageWords> side;

    std::shared_ptr<ShaderMemory> memory;
    std::vector<ShaderRecompiler::MemoryRegion> captured;
    std::vector<std::pair<std::uint64_t, std::size_t>> spans;
    std::atomic<std::uint64_t> generation{0};

    std::vector<std::uint32_t> userData;
    std::uint64_t addressWords = 0;
    std::uint64_t unusedWords = 0;
    std::vector<std::array<std::uint32_t, 3>> addressForwards;
    std::vector<DispatchAddressPosition> addressPositions;
    bool relocated = false;
    bool descriptorsPatched = false;
    bool writtenPatched = false;
    std::atomic<std::uint64_t> relocationProbe{0};
};

struct DispatchRelocation {
    std::uint64_t mask = 0;
    std::uint64_t aligned = 0;
    std::uint64_t banned = 0;
    std::uint32_t lookups = 0;
    std::uint32_t hits = 0;
    std::uint32_t idle = 0;
    std::vector<std::uint32_t> lastUserData;
    std::uint64_t lastRelocatable = 0;
    std::uint64_t lastAddresses = 0;
    std::uint64_t lastUnused = 0;
    std::uint64_t lastForwarded = 0;
    std::uint64_t lastValues = 0;
};

using DispatchAliasOf = std::function<std::uint64_t(const DispatchRelocation&)>;

struct DispatchProvenance {
    std::uint64_t identity = 0;
    std::span<const std::uint32_t> userData;
    std::vector<std::uint32_t> readNodes;
    std::shared_ptr<const ShaderRecompiler::SideForwards> sides;
    DispatchAliasOf aliasOf;
};

struct DispatchEntry {
    std::vector<std::shared_ptr<DispatchVariant>> variants;

    std::uint64_t touched = 0;

    std::list<std::uint64_t>::iterator order;
};

struct EntryCounters {
    std::uint64_t lookups = 0, absent = 0, equal = 0, differing = 0, queuedStores = 0, inaccessible = 0, queuedLabel = 0, flushingImage = 0, publishMoved = 0, pendingMoved = 0, forgetMoved = 0, deviceImageFallbacks = 0, imagesFlushed = 0, runsSynced = 0, forgetSinceInsert = 0, replaced = 0, inserts = 0, unstable = 0, touches = 0;

    std::uint64_t runsValidated = 0, runsInserted = 0, retriesEqual = 0, retriesMoved = 0;
    double validateUs = 0;

    std::uint64_t differingClassified = 0, differingAddress = 0, differingData = 0, differingWalk = 0, differingMixed = 0, differingRunsChanged = 0, differingMatchedPrior = 0, differingWords = 0;
    std::array<std::uint64_t, 4> differingWordBuckets{};

    std::array<std::uint64_t, MaxDispatchVariants> variantHitsByRank{};
    std::uint64_t variantsCompared = 0, variantsInserted = 0, variantsEvicted = 0;

    std::uint64_t dataHits = 0, dataWordsRefreshed = 0, dataVerified = 0, dataInserts = 0, dataPositionsInserted = 0, dataLeavesUnmapped = 0, dataLeavesMismatched = 0, dataLeavesAliased = 0;
    std::array<std::uint64_t, MaxDispatchVariants> dataHitsByRank{};
    std::set<std::size_t> differingPositions;
    std::size_t differingFirstPosition = std::numeric_limits<std::size_t>::max(), differingLastPosition = 0;
    std::map<std::uint64_t, std::uint64_t> differingByProgram;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

struct DataMask {
    std::span<const std::uint32_t> positions;
    std::vector<std::pair<std::uint32_t, std::uint32_t>>* live;
    std::span<const DispatchAddressPosition> addresses{};
};

enum class EntryOutcome { Equal, EqualData, Differing, QueuedStores, Inaccessible, QueuedLabel, FlushingImage, PublishMoved, PendingMoved, ForgetMoved, DeviceImageFallback };

void NoteDispatchLookup(EntryOutcome outcome, std::size_t compared);

bool CacheableResult(const ShaderRecompiler::RecompileResult& compiled);

}

#endif
