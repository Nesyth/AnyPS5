#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_STAGEPLANS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_STAGEPLANS_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRecords.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/StagePlanReport.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace AgcDriver::DriverDetail {

struct LeanShape;

struct StageRun {
    std::int16_t base = -1;
    std::uint64_t offset = 0;
    std::uint32_t words = 0;

    bool operator==(const StageRun&) const = default;
};

struct StageLayout {
    std::vector<StageRun> runs;
    std::vector<std::uint32_t> keyPositions;
    std::vector<std::uint32_t> bitPositions;
    std::uint64_t exactUsers = 0;
    std::uint64_t bitUsers = 0;
    std::uint32_t users = 0;
    std::uint32_t words = 0;

    bool operator==(const StageLayout&) const = default;
};

struct StagePlanKey {
    std::uint64_t hash = 0;
    std::vector<std::uint32_t> words;

    bool operator==(const StagePlanKey&) const = default;
};

struct StagePlan {
    std::shared_ptr<const StageLayout> layout;
    StagePlanKey key;
    DrawRecordStage stage;
    std::vector<std::uint32_t> vertexLayout;
    bool addressed = false;
    std::size_t bytes = 0;
};

struct StagePlanStage {
    std::vector<std::shared_ptr<const StageLayout>> layouts;
    std::vector<std::shared_ptr<const StagePlan>> plans;
};

struct DrawPlan {
    std::vector<std::shared_ptr<const StagePlan>> stages;
    std::shared_ptr<const DrawRecord> record;
    std::size_t bytes = 0;
};

struct StagePlanSet {
    std::shared_ptr<const LeanShape> shape;
    std::uint64_t generation = 0;
    std::uint64_t deviceSerial = 0;
    std::vector<StagePlanStage> stages;
    std::vector<DrawPlan> draws;
    std::size_t bytes = 0;

    std::size_t PlanCount() const;
};

struct StagePlanAnswer {
    std::shared_ptr<const LeanShape> shape;
    std::uint64_t generation = 0;
    std::uint64_t deviceSerial = 0;
    std::vector<std::shared_ptr<const StagePlan>> stages;
    std::shared_ptr<const DrawRecord> record;
    bool demote = false;
};

struct StagePlanMerge {
    std::size_t added = 0;
    std::size_t shared = 0;
    std::size_t replaced = 0;
    std::size_t evicted = 0;
    std::size_t layouts = 0;
    std::size_t drawsAdded = 0;
    std::size_t drawsKnown = 0;
    std::size_t drawsEvicted = 0;
};

namespace StagePlanLimits {

constexpr std::size_t Layouts = 4;
constexpr std::size_t PlansPerStage = 32;
constexpr std::size_t DrawPlans = 64;
constexpr std::uint64_t KeepPresents = 3;

}

std::shared_ptr<const StagePlan> MakeStagePlan(const DrawRecordStage& stage);
void StageKeyOf(const StageLayout& layout, std::span<const std::uint32_t> user, std::span<const std::uint32_t> words, StagePlanKey& key);
bool ReadStageRuns(const StageLayout& layout, std::span<const std::uint32_t> user, std::vector<std::uint32_t>& words);
const StagePlan* FindStagePlan(const StagePlanStage& stage, const StageLayout* layout, const StagePlanKey& key);
StagePlans::Counter StageKeyMiss(const StagePlanStage& stage, const StageLayout* layout, const StagePlanKey& key);
const DrawPlan* FindDrawPlan(const StagePlanSet& set, std::span<const StagePlan* const> stages);
std::shared_ptr<const DrawRecord> AssembleDrawRecord(const DrawRecord& built, std::span<const std::shared_ptr<const StagePlan>> stages);
std::shared_ptr<StagePlanSet> MergeStagePlans(const StagePlanSet* kept, const StagePlanAnswer& answer, StagePlanMerge& merge);

}

#endif
