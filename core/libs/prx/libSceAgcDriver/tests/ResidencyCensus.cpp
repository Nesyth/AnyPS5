#include "prx/libSceAgcDriver/Graphics/include/ResidencyCensus.hpp"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace Census = AgcDriver::Graphics::ResidencyCensus;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Census::UseFacts Valid() {
    Census::UseFacts facts;
    facts.seen = true;
    facts.sameBuild = true;
    facts.copiedBack = true;
    facts.armed = true;
    facts.layout = true;
    facts.covered = true;
    facts.sameImport = true;
    facts.watched = true;
    return facts;
}

struct Failure {
    void (*apply)(Census::UseFacts& facts);
    Census::Miss miss;
};

const std::vector<Failure>& Failures() {
    static const std::vector<Failure> failures{
        {[](Census::UseFacts& facts) { facts.copiedBack = false; }, Census::Miss::Lost},
        {[](Census::UseFacts& facts) { facts.armed = false; }, Census::Miss::Unarmed},
        {[](Census::UseFacts& facts) { facts.gap = true; }, Census::Miss::Gap},
        {[](Census::UseFacts& facts) { facts.layout = false; }, Census::Miss::Layout},
        {[](Census::UseFacts& facts) { facts.covered = false; }, Census::Miss::Uncovered},
        {[](Census::UseFacts& facts) { facts.sameImport = false; }, Census::Miss::Import},
        {[](Census::UseFacts& facts) { facts.watched = false; }, Census::Miss::Unwatched},
        {[](Census::UseFacts& facts) { facts.indexBreak = true; }, Census::Miss::IndexBreak},
        {[](Census::UseFacts& facts) { facts.bumped = true; }, Census::Miss::Bumped},
        {[](Census::UseFacts& facts) { facts.cpuAtMarks = true; }, Census::Miss::CpuAtMarks},
        {[](Census::UseFacts& facts) { facts.shadow = true; }, Census::Miss::Shadow},
        {[](Census::UseFacts& facts) { facts.pending = true; }, Census::Miss::Pending},
        {[](Census::UseFacts& facts) { facts.cpuWriter = true; }, Census::Miss::CpuWriter},
        {[](Census::UseFacts& facts) { facts.queuedStore = true; }, Census::Miss::QueuedStore},
    };
    return failures;
}

std::string Name(Census::Miss miss) {
    return Census::MissName(miss);
}

void CheckValidUseHits() {
    Require(Census::FirstMiss(Valid(), false) == Census::Miss::Hit, "a use that meets every condition misses one copy per build");
    Require(Census::FirstMiss(Valid(), true) == Census::Miss::Hit, "a use that meets every condition misses one copy per range");
    const Census::UseFacts never{};
    Require(Census::FirstMiss(never, false) == Census::Miss::First && Census::FirstMiss(never, true) == Census::Miss::First, "the first use of a range is not counted as first");
    auto rebuilt = Valid();
    rebuilt.sameBuild = false;
    Require(Census::FirstMiss(rebuilt, false) == Census::Miss::Rebuilt, "a use by another build hits one copy per build");
    Require(Census::FirstMiss(rebuilt, true) == Census::Miss::Hit, "a use by another build misses one copy per range although every other condition holds");
    auto first = rebuilt;
    first.seen = false;
    Require(Census::FirstMiss(first, false) == Census::Miss::First, "an unseen range reports a rebuilt build before the first use");
}

void CheckEachConditionAlone() {
    for (const auto& failure : Failures()) {
        auto facts = Valid();
        failure.apply(facts);
        Require(Census::FirstMiss(facts, false) == failure.miss, "one failing condition is reported as " + Name(Census::FirstMiss(facts, false)) + " instead of " + Name(failure.miss));
        Require(Census::FirstMiss(facts, true) == failure.miss, "one failing condition is reported per range as " + Name(Census::FirstMiss(facts, true)) + " instead of " + Name(failure.miss));
        auto rebuilt = facts;
        rebuilt.sameBuild = false;
        Require(Census::FirstMiss(rebuilt, false) == Census::Miss::Rebuilt, "another build with " + Name(failure.miss) + " is not reported as rebuilt first");
    }
}

void CheckEarlierConditionWins() {
    const auto& failures = Failures();
    for (std::size_t earlier = 0; earlier < failures.size(); ++earlier) {
        for (std::size_t later = earlier + 1u; later < failures.size(); ++later) {
            auto facts = Valid();
            failures[later].apply(facts);
            failures[earlier].apply(facts);
            Require(Census::FirstMiss(facts, false) == failures[earlier].miss, "with " + Name(failures[earlier].miss) + " and " + Name(failures[later].miss) + " failing, the census names " + Name(Census::FirstMiss(facts, false)));
        }
    }
}

void CheckMissNames() {
    std::set<std::string> names;
    for (std::size_t index = 0; index < static_cast<std::size_t>(Census::Miss::Count); ++index) {
        const auto name = Name(static_cast<Census::Miss>(index));
        Require(name != "unnamed" && !name.empty(), "miss " + std::to_string(index) + " has no name");
        Require(names.insert(name).second, "two misses share the name " + name);
    }
}

void CheckLineGeometry() {
    Require(Census::LinesSpanned(0, 0) == 0, "an empty range spans a line");
    Require(Census::LinesSpanned(0, 128) == 1, "one aligned line spans " + std::to_string(Census::LinesSpanned(0, 128)));
    Require(Census::LinesSpanned(0, 129) == 2, "a line and a byte span " + std::to_string(Census::LinesSpanned(0, 129)));
    Require(Census::LinesSpanned(124, 8) == 2, "eight bytes across a line boundary span " + std::to_string(Census::LinesSpanned(124, 8)));
    Require(Census::LinesSpanned(128 * 5 + 4, 124) == 1, "the rest of a line spans " + std::to_string(Census::LinesSpanned(128 * 5 + 4, 124)));
    Require(Census::LinesSpanned(4, 128) == 2, "a line's worth of bytes from the second dword spans " + std::to_string(Census::LinesSpanned(4, 128)));
    Require(Census::LinesSpanned(0x100000000ull, 0x780000ull) == 0x780000ull / 128u, "a 7.5 MiB range at a line boundary spans " + std::to_string(Census::LinesSpanned(0x100000000ull, 0x780000ull)) + " lines");
    Require(Census::LeadWords(0) == 0 && Census::LeadWords(4) == 1 && Census::LeadWords(124) == 31 && Census::LeadWords(128) == 0 && Census::LeadWords(0x100000044ull) == 17, "the lead dwords before a line boundary are wrong");
}

std::uint64_t ReferenceLines(std::uint32_t lead, const std::vector<std::uint32_t>& result, const std::vector<std::uint32_t>& snapshot) {
    std::set<std::uint64_t> lines;
    for (std::size_t index = 0; index < result.size(); ++index) {
        if (result[index] != snapshot[index]) lines.insert((std::uint64_t{lead} + index) / 32u);
    }
    return lines.size();
}

void CheckChangedLines() {
    constexpr std::uint32_t lead = 5;
    std::vector<std::uint32_t> snapshot(200);
    for (std::size_t index = 0; index < snapshot.size(); ++index) snapshot[index] = static_cast<std::uint32_t>(index * 2654435761u);
    auto result = snapshot;
    for (const std::size_t index : {std::size_t{0}, std::size_t{26}, std::size_t{27}, std::size_t{199}}) result[index] ^= 1u;
    for (std::size_t index = 100; index < 164; ++index) result[index] ^= 0x80000000u;
    const auto counted = Census::CountChangedLines(lead, result, snapshot);
    Require(counted.changedWords == 68, "the changed dwords are counted as " + std::to_string(counted.changedWords) + ", expected 68");
    Require(counted.changedLines == 6, "the changed lines are counted as " + std::to_string(counted.changedLines) + ", expected 6 (lines 0, 1, 3, 4, 5 and 6 counted from the lead)");
    const auto none = Census::CountChangedLines(lead, snapshot, snapshot);
    Require(none.changedLines == 0 && none.changedWords == 0, "identical bytes count as changed");
    for (std::uint32_t pattern = 1; pattern < 40; ++pattern) {
        for (const std::uint32_t shift : {0u, 1u, 17u, 31u}) {
            auto varied = snapshot;
            for (std::size_t index = 0; index < varied.size(); ++index) {
                if ((index * pattern + shift) % 37u < 3u) varied[index] += 1u;
            }
            const auto count = Census::CountChangedLines(shift, varied, snapshot);
            Require(count.changedLines == ReferenceLines(shift, varied, snapshot), "pattern " + std::to_string(pattern) + " with lead " + std::to_string(shift) + " counts " + std::to_string(count.changedLines) + " changed lines, the reference " + std::to_string(ReferenceLines(shift, varied, snapshot)));
        }
    }
}

void CheckDispatchPlan() {
    Require(Census::LineDispatches(0, 0, 65535).empty(), "an empty range is dispatched");
    Require(Census::LineDispatches(3, 100, 0).empty(), "a device without groups is dispatched");
    const auto single = Census::LineDispatches(5, 200, 65535);
    Require(single.size() == 1 && single.front().base == 0 && single.front().groups == 2, "205 positions are not one dispatch of two groups");
    for (const std::uint32_t lead : {0u, 7u, 31u}) {
        for (const std::uint32_t words : {1u, 127u, 128u, 1000u, 4096u, 4097u}) {
            for (const std::uint32_t groups : {1u, 3u, 8u}) {
                const auto plan = Census::LineDispatches(lead, words, groups);
                std::uint64_t covered = 0;
                for (const auto& step : plan) {
                    Require(step.base == covered, "a dispatch starts at " + std::to_string(step.base) + " instead of " + std::to_string(covered));
                    Require(step.base % Census::GroupWords == 0, "a dispatch starts inside a group of 128 positions");
                    Require(step.groups != 0 && step.groups <= groups, "a dispatch has " + std::to_string(step.groups) + " groups with a limit of " + std::to_string(groups));
                    covered += std::uint64_t{step.groups} * Census::GroupWords;
                }
                const auto total = std::uint64_t{lead} + words;
                Require(covered >= total && covered < total + Census::GroupWords, "lead " + std::to_string(lead) + " and " + std::to_string(words) + " dwords are covered by " + std::to_string(covered) + " positions");
            }
        }
    }
}

void CheckTags() {
    const auto tag = Census::Tag(Census::Stamper::StagedMark, 3);
    Require((tag >> 56u) == 16u && (tag & Census::TagWhoMask) == 3u, "a staged mark tag does not keep its kind and region");
    Require(std::string(Census::StamperName(tag)) == "staged-mark", "a staged mark tag is named " + std::string(Census::StamperName(tag)));
    Require(Census::Tag(Census::Stamper::Dispatch, 0x123456789abcull) == ((std::uint64_t{1} << 56u) | 0x123456789abcull), "a dispatch tag differs from the recorder's writer tag");
    Require(std::string(Census::StamperName(0)) == "untagged", "an untagged stamp is named " + std::string(Census::StamperName(0)));
}

}

int main() {
    try {
        CheckValidUseHits();
        CheckEachConditionAlone();
        CheckEarlierConditionWins();
        CheckMissNames();
        CheckLineGeometry();
        CheckChangedLines();
        CheckDispatchPlan();
        CheckTags();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "residency census tests passed\n";
    return 0;
}
