#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>

namespace AgcDriver::DriverDetail {

namespace {

enum RelocationOutcome : std::size_t { RelocatedHit, RelocatedDescriptors, RelocatedPush, RelocatedDiffering, RelocatedRefused, RelocatedNoShape, RelocatedNoAlias, RelocatedIdle, RelocatedExactPatched, RelocationOutcomes };
enum MissWord : std::size_t { MissBase, MissAddress, MissUnused, MissForwarded, MissValue, MissOther, MissWords };

struct RelocationCensus {
    std::array<std::atomic<std::uint64_t>, RelocationOutcomes> outcomes{};
    std::array<std::atomic<std::uint64_t>, MissWords> words{};
    std::array<std::atomic<std::uint64_t>, RelocationOutcomes> trials{};
    std::atomic<std::uint64_t> validated{0}, skipped{0}, learned{0}, skippedAll{0};
    std::atomic<std::int64_t> lastReport{0};
};

RelocationCensus& Relocations() {
    static RelocationCensus census;
    return census;
}

void reportRelocations() {
    auto& census = Relocations();
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = census.lastReport.load();
    if (last == 0) {
        census.lastReport.compare_exchange_strong(last, nowMs);
        return;
    }
    if (nowMs - last < 10000 || !census.lastReport.compare_exchange_strong(last, nowMs)) return;
    std::array<unsigned long long, RelocationOutcomes> outcome{};
    for (std::size_t index = 0; index < outcome.size(); ++index) outcome[index] = static_cast<unsigned long long>(census.outcomes[index].exchange(0, std::memory_order_relaxed));
    std::array<unsigned long long, MissWords> word{};
    for (std::size_t index = 0; index < word.size(); ++index) word[index] = static_cast<unsigned long long>(census.words[index].exchange(0, std::memory_order_relaxed));
    const auto misses = outcome[RelocatedHit] + outcome[RelocatedDiffering] + outcome[RelocatedRefused] + outcome[RelocatedNoShape] + outcome[RelocatedNoAlias] + outcome[RelocatedIdle];
    std::fprintf(stderr, "[dispatch-relocate] (10 s) %llu exact misses: %llu relocated hits (%llu with patched descriptors, %llu with patched push words), %llu relocated but differing, %llu with moves the donors refuse, %llu with no shape, %llu with no alias, %llu with the trial idle; misses by the user words that differ from the program's last insert: base %llu, buffer address %llu, unused %llu, forwarded %llu, walked value %llu, other %llu; %llu exact hits served through a patched copy\n", misses, outcome[RelocatedHit], outcome[RelocatedDescriptors], outcome[RelocatedPush], outcome[RelocatedDiffering], outcome[RelocatedRefused], outcome[RelocatedNoShape], outcome[RelocatedNoAlias], outcome[RelocatedIdle], word[MissBase], word[MissAddress], word[MissUnused], word[MissForwarded], word[MissValue], word[MissOther], outcome[RelocatedExactPatched]);
    std::array<unsigned long long, RelocationOutcomes> trial{};
    for (std::size_t index = 0; index < trial.size(); ++index) trial[index] = static_cast<unsigned long long>(census.trials[index].exchange(0, std::memory_order_relaxed));
    const auto validated = static_cast<unsigned long long>(census.validated.exchange(0, std::memory_order_relaxed));
    const auto skipped = static_cast<unsigned long long>(census.skipped.exchange(0, std::memory_order_relaxed));
    const auto learned = static_cast<unsigned long long>(census.learned.exchange(0, std::memory_order_relaxed));
    const auto skippedAll = static_cast<unsigned long long>(census.skippedAll.exchange(0, std::memory_order_relaxed));
    const auto tried = trial[RelocatedHit] + trial[RelocatedDiffering] + trial[RelocatedRefused] + trial[RelocatedNoShape] + trial[RelocatedNoAlias];
    if (tried != 0) std::fprintf(stderr, "[dispatch-relocate-trials] (10 s) %llu exact misses tried with no idle window: %llu relocated hits; not served: %llu with no shape, %llu with no alias, %llu with moves the donors refuse, %llu differing (%llu with every donor skipped by its remembered word); donor variants: %llu validated, %llu skipped by a remembered differing word, %llu differing words learned\n", tried, trial[RelocatedHit], trial[RelocatedNoShape], trial[RelocatedNoAlias], trial[RelocatedRefused], trial[RelocatedDiffering], skippedAll, validated, skipped, learned);
}

bool ElementWritten(const ShaderRecompiler::RecompileResult& compiled, std::uint32_t binding, std::uint32_t word) {
    if (binding >= compiled.bindings.size()) return true;
    const auto& target = compiled.bindings[binding];
    const auto element = word / 4u;
    return element >= target.bufferWritten.size() || target.bufferWritten[element];
}

struct TrialVerify {
    std::atomic<std::uint64_t> seen{0}, checked{0}, differing{0}, printed{0};
};

TrialVerify& TrialChecks() {
    static TrialVerify verify;
    return verify;
}

std::uint64_t VerifyTrialsEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_DISPATCH_RELOCATE_TRIALS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

bool RelocationWordAccepted(const DispatchVariant& variant, std::uint32_t position, std::uint32_t stored, std::uint32_t fresh) {
    if (stored == fresh || std::binary_search(variant.dataPositions.begin(), variant.dataPositions.end(), position)) return true;
    const auto found = std::lower_bound(variant.addressPositions.begin(), variant.addressPositions.end(), position, [](const DispatchAddressPosition& entry, std::uint32_t value) { return entry.position < value; });
    return found != variant.addressPositions.end() && found->position == position && ((stored ^ fresh) & ~found->mask) == 0 && !(found->guarded && (fresh & found->mask) == 0);
}

std::uint64_t RelocationProbeOf(const DispatchVariant& copy, std::span<const ShaderRecompiler::MemoryRegion> regions) {
    struct LiveStorage {};
    auto& live = HostThreadLocal<std::vector<std::byte>, LiveStorage>();
    std::size_t first = 0;
    for (std::size_t run = 0; run < regions.size(); ++run) {
        const auto& region = regions[run];
        const auto count = region.bytes.size() / sizeof(std::uint32_t);
        live.resize(region.bytes.size());
        if (GuestMemory::CopyMapped(region.guestAddress, live) == GuestMemory::Compare::Equal) {
            for (std::size_t word = 0; word < count; ++word) {
                std::uint32_t stored = 0;
                std::uint32_t fresh = 0;
                std::memcpy(&stored, region.bytes.data() + word * sizeof(std::uint32_t), sizeof(stored));
                std::memcpy(&fresh, live.data() + word * sizeof(std::uint32_t), sizeof(fresh));
                if (!RelocationWordAccepted(copy, static_cast<std::uint32_t>(first + word), stored, fresh)) return (static_cast<std::uint64_t>(run + 1u) << 32u) | static_cast<std::uint64_t>(word);
            }
        }
        first += count;
    }
    return 0;
}

bool RelocationProbeDiffers(const DispatchVariant& donor, std::span<const std::uint32_t> userData, std::uint64_t probe) {
    const auto run = static_cast<std::size_t>((probe >> 32u) - 1u);
    const auto word = static_cast<std::size_t>(probe & 0xffffffffu);
    if (donor.compiled == nullptr || donor.userData.size() != userData.size() || run >= donor.runs.size()) return false;
    std::uint64_t moves = 0;
    for (std::size_t index = 0; index < userData.size(); ++index) {
        const auto from = donor.userData[index];
        const auto to = userData[index];
        if (from == to) continue;
        if (index >= 64) return false;
        const auto bit = std::uint64_t{1} << index;
        if (((donor.relocatableWords | donor.addressWords) & bit) != 0) {
            if (from == 0 || to == 0 || ((from ^ to) & 3u) != 0) return false;
        } else if ((donor.unusedWords & bit) == 0) {
            return false;
        }
        moves |= bit;
    }
    if (moves == 0) return false;
    struct RunsStorage {};
    auto& runs = HostThreadLocal<std::vector<std::pair<std::uint64_t, std::uint64_t>>, RunsStorage>();
    runs.assign(donor.runs.begin(), donor.runs.end());
    for (std::uint32_t index = 0; index < 64 && index < userData.size(); ++index) {
        if ((moves >> index & 1u) == 0 || (donor.relocatableWords >> index & 1u) == 0) continue;
        const auto delta = static_cast<std::uint64_t>(userData[index] & ~3u) - static_cast<std::uint64_t>(donor.userData[index] & ~3u);
        for (std::size_t at = 0; at < runs.size() && at < donor.runBases.size(); ++at) {
            if (donor.runBases[at] != static_cast<std::int8_t>(index)) continue;
            runs[at].first += delta;
            runs[at].second += delta;
        }
    }
    std::size_t position = word;
    for (std::size_t at = 0; at < run; ++at) position += static_cast<std::size_t>((runs[at].second - runs[at].first) / sizeof(std::uint32_t));
    if (word >= static_cast<std::size_t>((runs[run].second - runs[run].first) / sizeof(std::uint32_t)) || position >= donor.words.size()) return false;
    if (DrawPipeline::Active()) {
        auto& pipeline = DrawPipeline::Queue0();
        if (pipeline.Busy() && pipeline.OverlapsAny(runs)) return false;
    }
    PendingView pending;
    pending.Load();
    for (const auto& [begin, end] : runs) {
        if (pending.Overlaps(begin, static_cast<std::size_t>(end - begin))) return false;
    }
    if (Graphics::StorageTexture::AnyPendingOverlaps(runs) || Graphics::AnyShadowedOverlaps(runs)) return false;
    std::array<std::byte, sizeof(std::uint32_t)> live{};
    if (GuestMemory::CopyMapped(runs[run].first + word * sizeof(std::uint32_t), live) != GuestMemory::Compare::Equal) return false;
    std::uint32_t fresh = 0;
    std::memcpy(&fresh, live.data(), sizeof(fresh));
    return !RelocationWordAccepted(donor, static_cast<std::uint32_t>(position), donor.words[position], fresh);
}

}

bool Driver::dispatchCensusEnabled() {
    static const bool census = std::getenv("APS5_DISPATCH_CENSUS") != nullptr;
    return census;
}

void Driver::noteDispatchRelocation(std::size_t outcome, std::uint64_t identity, std::span<const std::uint32_t> userData) {
    if (!dispatchCensusEnabled()) return;
    auto& census = Relocations();
    census.outcomes[outcome].fetch_add(1, std::memory_order_relaxed);
    if (outcome != RelocatedHit && outcome != RelocatedDescriptors && outcome != RelocatedPush && outcome != RelocatedExactPatched) {
        std::array<bool, MissWords> seen{};
        {
            std::lock_guard cacheLock(dispatchCacheMutex);
            const auto found = dispatchRelocations.find(identity);
            if (found == dispatchRelocations.end() || found->second.lastUserData.size() != userData.size()) {
                seen[MissOther] = true;
            } else {
                const auto& shape = found->second;
                for (std::size_t index = 0; index < userData.size(); ++index) {
                    if (shape.lastUserData[index] == userData[index]) continue;
                    const auto bit = index < 64 ? std::uint64_t{1} << index : std::uint64_t{0};
                    const auto kind = (shape.lastAddresses & bit) != 0 ? MissAddress : (shape.lastRelocatable & bit) != 0 ? MissBase : (shape.lastUnused & bit) != 0 ? MissUnused : (shape.lastForwarded & bit) != 0 ? MissForwarded : (shape.lastValues & bit) != 0 ? MissValue : MissOther;
                    seen[kind] = true;
                }
            }
        }
        for (std::size_t kind = 0; kind < MissWords; ++kind) {
            if (seen[kind]) census.words[kind].fetch_add(1, std::memory_order_relaxed);
        }
    }
    reportRelocations();
}

std::shared_ptr<DispatchVariant> Driver::relocatedCopy(const std::shared_ptr<DispatchVariant>& variant, std::span<const std::uint32_t> userData, bool requireMoves) {
    const auto& donor = *variant;
    if (donor.compiled == nullptr || donor.userData.size() != userData.size()) return nullptr;
    std::uint64_t moves = 0;
    for (std::size_t index = 0; index < userData.size(); ++index) {
        const auto from = donor.userData[index];
        const auto to = userData[index];
        if (from == to) continue;
        if (index >= 64) return nullptr;
        const auto bit = std::uint64_t{1} << index;
        if (((donor.relocatableWords | donor.addressWords) & bit) != 0) {
            if (from == 0 || to == 0 || ((from ^ to) & 3u) != 0) return nullptr;
        } else if ((donor.unusedWords & bit) == 0) {
            return nullptr;
        }
        moves |= bit;
    }
    if (moves == 0 && requireMoves) return nullptr;
    auto copy = std::make_shared<DispatchVariant>();
    copy->runs = donor.runs;
    copy->runBases = donor.runBases;
    copy->words = donor.words;
    copy->forgetSerial = donor.forgetSerial;
    copy->compiled = donor.compiled;
    copy->registeredViews = donor.registeredViews;
    copy->shader = donor.shader;
    copy->dataPositions = donor.dataPositions;
    copy->dataSlots = donor.dataSlots;
    copy->flatBinding = donor.flatBinding;
    copy->pushOffset = donor.pushOffset;
    copy->relocatableWords = donor.relocatableWords;
    copy->userWordsWalked = donor.userWordsWalked;
    copy->addressWords = donor.addressWords;
    copy->unusedWords = donor.unusedWords;
    copy->addressForwards = donor.addressForwards;
    copy->addressPositions = donor.addressPositions;
    copy->userData.assign(userData.begin(), userData.end());
    copy->donor = donor.donor != nullptr ? donor.donor : variant;
    copy->relocated = true;
    const auto& compiled = *donor.compiled;
    for (std::uint32_t index = 0; index < 64 && index < userData.size(); ++index) {
        if ((moves >> index & 1u) == 0) continue;
        const auto from = donor.userData[index];
        const auto to = userData[index];
        if ((donor.relocatableWords >> index & 1u) != 0) {
            const auto delta = static_cast<std::uint64_t>(to & ~3u) - static_cast<std::uint64_t>(from & ~3u);
            for (std::size_t run = 0; run < copy->runs.size() && run < copy->runBases.size(); ++run) {
                if (copy->runBases[run] != static_cast<std::int8_t>(index)) continue;
                copy->runs[run].first += delta;
                copy->runs[run].second += delta;
            }
        }
        for (std::uint32_t slot = 0; slot < compiled.shaderDataUserWords.size(); ++slot) {
            if (compiled.shaderDataUserWords[slot] == index) copy->shaderDataPatches.emplace_back(slot, to);
        }
        for (const auto& [word, binding, position] : donor.addressForwards) {
            if (word != index) continue;
            copy->descriptorPatches.push_back({binding, position, to});
            copy->descriptorsPatched = true;
            copy->writtenPatched = copy->writtenPatched || ElementWritten(compiled, binding, position);
        }
    }
    return copy;
}

std::shared_ptr<const ShaderRecompiler::RecompileResult> Driver::patchRelocated(DispatchVariant& copy, const DispatchVariant& donor, std::vector<std::uint32_t>& liveWords, std::span<const std::pair<std::uint32_t, std::uint32_t>> live) {
    auto result = std::make_shared<ShaderRecompiler::RecompileResult>(*donor.compiled);
    for (const auto& [slot, value] : copy.shaderDataPatches) patchForwardedWord(*result, slot, value);
    for (const auto& [binding, word, value] : copy.descriptorPatches) {
        if (binding < result->bindings.size() && word < result->bindings[binding].guestDescriptor.size()) result->bindings[binding].guestDescriptor[word] = value;
    }
    if (!live.empty() && copy.flatBinding < result->bindings.size()) {
        auto& descriptor = result->bindings[copy.flatBinding].guestDescriptor;
        for (std::size_t k = 0; k < copy.dataPositions.size() && k < copy.dataSlots.size(); ++k) {
            if (copy.dataSlots[k] < descriptor.size() && copy.dataPositions[k] < liveWords.size()) descriptor[copy.dataSlots[k]] = liveWords[copy.dataPositions[k]];
        }
    }
    for (const auto& [position, value] : live) {
        for (const auto& entry : copy.addressPositions) {
            if (entry.position != position || entry.binding >= result->bindings.size() || entry.word >= result->bindings[entry.binding].guestDescriptor.size()) continue;
            auto& word = result->bindings[entry.binding].guestDescriptor[entry.word];
            const auto patched = (word & ~entry.mask) | (value & entry.mask);
            if (patched == word) continue;
            word = patched;
            copy.descriptorsPatched = true;
            copy.writtenPatched = copy.writtenPatched || ElementWritten(*donor.compiled, entry.binding, entry.word);
        }
    }
    return result;
}

void Driver::inheritRelocatedRecipe(DispatchVariant& copy, const DispatchVariant& donor) {
    const auto recipe = donor.recipe.load(std::memory_order_acquire);
    if (recipe == nullptr || copy.compiled == nullptr || donor.compiled == nullptr) return;
    const bool plain = !copy.descriptorsPatched && copy.compiled->pushConstants == donor.compiled->pushConstants;
    const bool moves = recipe->moves && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DispatchMovedRecipes) && (!copy.writtenPatched || Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DispatchMovedWrites));
    if (plain || moves) copy.recipe.store(recipe, std::memory_order_release);
}

bool Driver::relocateDispatch(std::uint64_t address, const Submission& submission, std::uint64_t identity, const DispatchAliasOf& aliasOf, const std::shared_ptr<const ShaderSnapshot>& registeredShader, std::span<const std::uint32_t> userData, std::span<const ShaderRecompiler::MemoryRegion> memory, std::shared_ptr<const ShaderRecompiler::RecompileResult>& compiledResult, std::shared_ptr<DispatchVariant>& keepVariant, std::vector<ShaderRecompiler::MemoryRegion>& captured, std::vector<std::uint32_t>& liveWords) {
    constexpr std::uint32_t TrialWindow = 128;
    constexpr std::uint32_t TrialRest = 2048;
    const bool trials = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DispatchRelocateTrials);
    std::vector<std::pair<std::uint64_t, std::shared_ptr<DispatchEntry>>> donors;
    auto outcome = RelocatedNoAlias;
    {
        std::lock_guard cacheLock(dispatchCacheMutex);
        const auto shaped = dispatchRelocations.find(identity);
        if (shaped == dispatchRelocations.end()) {
            outcome = RelocatedNoShape;
        } else if (!trials && shaped->second.idle != 0) {
            --shaped->second.idle;
            outcome = RelocatedIdle;
        } else if (const auto found = dispatchAliases.find(aliasOf(shaped->second)); found != dispatchAliases.end()) {
            for (auto& key : found->second) {
                if (key == 0) continue;
                const auto entry = dispatchCache.find(key);
                if (entry == dispatchCache.end()) {
                    key = 0;
                    continue;
                }
                donors.emplace_back(key, entry->second);
            }
            std::stable_partition(found->second.begin(), found->second.end(), [](std::uint64_t key) { return key != 0; });
            if (donors.empty()) dispatchAliases.erase(found);
        }
    }
    if (donors.empty()) {
        if (outcome == RelocatedNoAlias && !trials) {
            std::lock_guard cacheLock(dispatchCacheMutex);
            if (const auto shaped = dispatchRelocations.find(identity); shaped != dispatchRelocations.end() && ++shaped->second.lookups >= TrialWindow) {
                if (shaped->second.hits * 8u < shaped->second.lookups) shaped->second.idle = TrialRest;
                shaped->second.lookups = 0;
                shaped->second.hits = 0;
            }
        }
        if (trials && dispatchCensusEnabled()) Relocations().trials[outcome].fetch_add(1, std::memory_order_relaxed);
        noteDispatchRelocation(outcome, identity, userData);
        return false;
    }
    outcome = RelocatedRefused;
    std::optional<SampledReadScope> sampling;
    std::uint64_t imagesFlushed = 0, runsSynced = 0;
    std::vector<ShaderRecompiler::MemoryRegion> regions;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> liveData;
    std::shared_ptr<DispatchVariant> hit;
    std::shared_ptr<DispatchVariant> hitDonor;
    std::uint64_t hitKey = 0;
    std::shared_ptr<DispatchEntry> hitEntry;
    const bool trialCensus = trials && dispatchCensusEnabled();
    std::uint64_t validated = 0, skipped = 0;
    for (const auto& [key, entry] : donors) {
        for (const auto& variant : entry->variants) {
            if (variant == nullptr || variant->shader != registeredShader) continue;
            if (variant->registeredViews != nullptr && !CaptureRegisteredViewsCurrent(*variant->registeredViews)) continue;
            if (trials) {
                const auto probe = variant->relocationProbe.load(std::memory_order_relaxed);
                if (probe != 0 && RelocationProbeDiffers(*variant, userData, probe)) {
                    ++skipped;
                    outcome = RelocatedDiffering;
                    auto& verify = TrialChecks();
                    if (VerifyTrialsEvery() != 0 && verify.seen.fetch_add(1, std::memory_order_relaxed) % VerifyTrialsEvery() == 0) {
                        bool validates = false;
                        try {
                            if (const auto check = relocatedCopy(variant, userData); check != nullptr) {
                                std::vector<ShaderRecompiler::MemoryRegion> checkRegions;
                                appendEntryRegions(*check, checkRegions);
                                std::vector<std::pair<std::uint32_t, std::uint32_t>> checkData;
                                std::optional<SampledReadScope> checkSampling;
                                std::uint64_t checkImages = 0, checkRuns = 0;
                                const GuestMemory::ReadSiteScope checkSite(GuestMemory::ReadSite::DispatchCache);
                                const auto checked = validateVariant(address, submission.queue, *check, checkRegions, checkImages, checkRuns, checkSampling, &checkData, true);
                                validates = checked == EntryOutcome::Equal || checked == EntryOutcome::EqualData;
                            }
                        } catch (const std::exception& error) {
                            if (verify.printed.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[dispatch-relocate-trials-verify] program 0x%llx: the check could not be made: %s\n", static_cast<unsigned long long>(address), error.what());
                        }
                        const auto count = verify.checked.fetch_add(1, std::memory_order_relaxed) + 1;
                        if (validates) {
                            verify.differing.fetch_add(1, std::memory_order_relaxed);
                            if (verify.printed.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[dispatch-relocate-trials-verify] program 0x%llx: a donor skipped by its remembered word (run %llu, word %llu) validates\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>((probe >> 32u) - 1u), static_cast<unsigned long long>(probe & 0xffffffffu));
                        }
                        if (count % 4096 == 0) std::fprintf(stderr, "[dispatch-relocate-trials-verify] %llu checked, %llu differing\n", static_cast<unsigned long long>(count), static_cast<unsigned long long>(verify.differing.load(std::memory_order_relaxed)));
                    }
                    continue;
                }
            }
            auto copy = relocatedCopy(variant, userData);
            if (copy == nullptr) continue;
            regions.clear();
            appendEntryRegions(*copy, regions);
            const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DispatchCache);
            const auto result = validateVariant(address, submission.queue, *copy, regions, imagesFlushed, runsSynced, sampling, &liveData, true);
            ++validated;
            if (result != EntryOutcome::Equal && result != EntryOutcome::EqualData) {
                outcome = RelocatedDiffering;
                if (trials && result == EntryOutcome::Differing) {
                    if (const auto learned = RelocationProbeOf(*copy, regions); learned != 0) {
                        variant->relocationProbe.store(learned, std::memory_order_relaxed);
                        if (trialCensus) Relocations().learned.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                continue;
            }
            if (trials && variant->relocationProbe.load(std::memory_order_relaxed) != 0) variant->relocationProbe.store(0, std::memory_order_relaxed);
            hit = std::move(copy);
            hitDonor = variant;
            hitKey = key;
            hitEntry = entry;
            break;
        }
        if (hit != nullptr) break;
    }
    if (hit != nullptr) {
        liveWords = hit->words;
        for (const auto& [position, value] : liveData) liveWords[position] = value;
        regions.clear();
        appendEntryRegions(*hit, regions, &liveWords);
        compiledResult = patchRelocated(*hit, *hitDonor, liveWords, liveData);
        hit->compiled = compiledResult;
        inheritRelocatedRecipe(*hit, *hitDonor);
        captured.reserve(memory.size() + regions.size());
        captured.assign(memory.begin(), memory.end());
        captured.insert(captured.end(), regions.begin(), regions.end());
        const bool pushPatched = compiledResult->pushConstants != hitDonor->compiled->pushConstants;
        const bool descriptorsPatched = hit->descriptorsPatched;
        keepVariant = std::move(hit);
        outcome = RelocatedHit;
        {
            std::lock_guard cacheLock(dispatchCacheMutex);
            ++dispatchCacheHits;
            if (const auto again = dispatchCache.find(hitKey); again != dispatchCache.end() && again->second == hitEntry && dispatchCacheHits - hitEntry->touched > dispatchCacheEntries() / 8) {
                dispatchOrder.splice(dispatchOrder.begin(), dispatchOrder, hitEntry->order);
                hitEntry->touched = dispatchCacheHits;
            }
        }
        if (descriptorsPatched) noteDispatchRelocation(RelocatedDescriptors, identity, userData);
        if (pushPatched) noteDispatchRelocation(RelocatedPush, identity, userData);
    }
    if (!trials) {
        std::lock_guard cacheLock(dispatchCacheMutex);
        if (const auto shaped = dispatchRelocations.find(identity); shaped != dispatchRelocations.end()) {
            if (outcome == RelocatedHit) ++shaped->second.hits;
            if (++shaped->second.lookups >= TrialWindow) {
                if (shaped->second.hits * 8u < shaped->second.lookups) shaped->second.idle = TrialRest;
                shaped->second.lookups = 0;
                shaped->second.hits = 0;
            }
        }
    }
    if (trialCensus) {
        auto& census = Relocations();
        census.trials[outcome].fetch_add(1, std::memory_order_relaxed);
        census.validated.fetch_add(validated, std::memory_order_relaxed);
        census.skipped.fetch_add(skipped, std::memory_order_relaxed);
        if (outcome != RelocatedHit && validated == 0 && skipped != 0) census.skippedAll.fetch_add(1, std::memory_order_relaxed);
    }
    noteDispatchRelocation(outcome, identity, userData);
    return outcome == RelocatedHit;
}

void Driver::lookupDispatch(std::uint64_t address, const Submission& submission, std::uint64_t key, bool noDispatchCache, bool traceCache, bool profile, std::span<const ShaderRecompiler::MemoryRegion> memory, DispatchPhaseTiming& phaseTiming, std::array<double, DriverPhaseCount>& phaseMs, std::shared_ptr<const ShaderRecompiler::RecompileResult>& compiledResult, std::shared_ptr<DispatchVariant>& keepVariant, std::vector<ShaderRecompiler::MemoryRegion>& captured, std::vector<std::uint32_t>& liveWords, bool& dataHit, bool& cached, bool& validated, std::shared_ptr<DispatchEntry>& missedEntry, bool& missedDiffering, bool relocate) {
    if (!noDispatchCache) {

        static const bool validateUnlocked = std::getenv("APS5_NO_UNLOCKED_VALIDATE") == nullptr;
        std::unique_lock cacheLock(dispatchCacheMutex);
        ++entryCounters.lookups;
        const auto found = dispatchCache.find(key);
        std::shared_ptr<DispatchEntry> entry = found != dispatchCache.end() ? found->second : nullptr;
        if (entry == nullptr) ++entryCounters.absent;
        if (validateUnlocked) cacheLock.unlock();
        phaseTiming.Phase(PhaseLookup);
        if (entry != nullptr) {
            validated = true;
            const auto& variants = entry->variants;

            std::shared_ptr<DispatchVariant> variant;
            std::size_t rank = 0;
            const auto generation = variants.front()->generation.load(std::memory_order_acquire);
            bool current = false;
            std::uint64_t restamped = 0;

            auto outcome = EntryOutcome::Differing;
            std::uint64_t imagesFlushed = 0, runsSynced = 0, retriesEqual = 0, retriesMoved = 0, compared = 0;
            std::vector<ShaderRecompiler::MemoryRegion> regions;

            std::vector<std::pair<std::uint32_t, std::uint32_t>> liveData;
            const auto waitedBeforeValidate = profile ? Graphics::Recorder::ThreadWaitedMs() : 0.0;
            if (!stampValidate() || variants.front()->linking != nullptr) {
                const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DispatchCache);

                std::optional<SampledReadScope> sampling;

                for (std::size_t i = 0; i < variants.size(); ++i) {
                    if (variants[i]->registeredViews != nullptr && !CaptureRegisteredViewsCurrent(*variants[i]->registeredViews)) continue;
                    regions.clear();
                    appendEntryRegions(*variants[i], regions);
                    ++compared;
                    auto result = validateVariant(address, submission.queue, *variants[i], regions, imagesFlushed, runsSynced, sampling, &liveData, relocate);

                    if (gateRetry() && (result == EntryOutcome::PublishMoved || result == EntryOutcome::PendingMoved)) {
                        result = validateVariant(address, submission.queue, *variants[i], regions, imagesFlushed, runsSynced, sampling, &liveData, relocate);
                        ++(result == EntryOutcome::Equal || result == EntryOutcome::EqualData ? retriesEqual : retriesMoved);
                    }
                    if (i == 0) outcome = result;
                    if (result == EntryOutcome::Equal || result == EntryOutcome::EqualData) {
                        outcome = result;
                        variant = variants[i];
                        rank = i;
                        dataHit = result == EntryOutcome::EqualData;
                        break;
                    }
                }
                current = variant != nullptr;
            } else {
                variant = variants.front();
                current = variant->registeredViews == nullptr || CaptureRegisteredViewsCurrent(*variant->registeredViews);
                if (!current) variant.reset();
                if (variant != nullptr) {
                    compared = 1;
                    current = true;
                    for (const auto& [begin, bytes] : variant->spans) {
                        GuestMemory::CollectWrites(begin, bytes);
                        if (!GuestMemory::UnchangedSince(begin, bytes, generation)) {
                            current = false;
                            break;
                        }
                    }
                }
                if (!current && variant != nullptr) {

                    std::uint64_t collected = 0;
                    for (const auto& [begin, bytes] : variant->spans) collected = std::max(collected, GuestMemory::CollectWrites(begin, bytes));
                    bool same = collected != 0;
                    if (same) {
                        const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DispatchCache);
                        PendingView pending;
                        pending.Load();
                        same = validateCaptured(address, submission.queue, variant->captured, *variant->compiled, false, pending);
                    }
                    if (same) {
                        restamped = collected;
                        current = true;
                    }
                }
                if (current) outcome = EntryOutcome::Equal;
            }
            phaseTiming.Phase(PhaseValidate);
            if (profile) {
                const auto waited = std::min(Graphics::Recorder::ThreadWaitedMs() - waitedBeforeValidate, phaseMs[PhaseValidate]);
                phaseMs[PhaseValidate] -= waited;
                phaseMs[PhaseValidateWait] += waited;
            }
            if (!cacheLock.owns_lock()) cacheLock.lock();

            const auto again = dispatchCache.find(key);
            const bool untouched = again != dispatchCache.end() && again->second == entry && variants.front()->generation.load(std::memory_order_acquire) == generation;
            auto& counters = entryCounters;
            counters.validateUs += phaseMs[PhaseValidate] * 1000;
            counters.imagesFlushed += imagesFlushed;
            counters.runsSynced += runsSynced;
            for (std::size_t i = 0; i < compared && i < variants.size(); ++i) counters.runsValidated += variants[i]->runs.size();
            counters.retriesEqual += retriesEqual;
            counters.retriesMoved += retriesMoved;
            counters.variantsCompared += compared;
            NoteDispatchLookup(outcome, compared);
            switch (outcome) {
                case EntryOutcome::Equal: ++counters.equal; break;
                case EntryOutcome::EqualData: ++counters.equal; break;
                case EntryOutcome::Differing: ++counters.differing; break;
                case EntryOutcome::QueuedStores: ++counters.queuedStores; break;
                case EntryOutcome::Inaccessible: ++counters.inaccessible; break;
                case EntryOutcome::QueuedLabel: ++counters.queuedLabel; break;
                case EntryOutcome::FlushingImage: ++counters.flushingImage; break;
                case EntryOutcome::PublishMoved: ++counters.publishMoved; break;
                case EntryOutcome::PendingMoved: ++counters.pendingMoved; break;
                case EntryOutcome::ForgetMoved: ++counters.forgetMoved; break;
                case EntryOutcome::DeviceImageFallback: ++counters.deviceImageFallbacks; break;
            }
            std::shared_ptr<DispatchVariant> patchedCopy;
            if (current) {
                ++counters.variantHitsByRank[rank];
                compiledResult = variant->compiled;
                if (dataHit) {

                    liveWords = variant->words;
                    for (const auto& [position, value] : liveData) liveWords[position] = value;
                    const bool addressMoved = relocate && !variant->addressPositions.empty() && std::any_of(liveData.begin(), liveData.end(), [&](const std::pair<std::uint32_t, std::uint32_t>& entry) { return std::any_of(variant->addressPositions.begin(), variant->addressPositions.end(), [&](const DispatchAddressPosition& held) { return held.position == entry.first; }); });
                    if (addressMoved) patchedCopy = relocatedCopy(variant, variant->userData, false);
                    regions.clear();
                    appendEntryRegions(patchedCopy != nullptr ? *patchedCopy : *variant, regions, &liveWords);
                    if (patchedCopy != nullptr) {
                        compiledResult = patchRelocated(*patchedCopy, *variant, liveWords, liveData);
                        patchedCopy->compiled = compiledResult;
                        inheritRelocatedRecipe(*patchedCopy, *variant);
                        noteDispatchRelocation(RelocatedExactPatched, 0, {});
                    } else {
                        compiledResult = patchDataWords(*variant, liveWords);
                    }
                    ++counters.dataHits;
                    counters.dataWordsRefreshed += liveData.size();
                    ++counters.dataHitsByRank[rank];
                }
                if (stampValidate() && variant->linking == nullptr) {
                    captured = variant->captured;
                } else {

                    captured.reserve(memory.size() + regions.size());
                    captured.assign(memory.begin(), memory.end());
                    captured.insert(captured.end(), regions.begin(), regions.end());
                    if (variant->forgetSerial != GuestMemory::ForgetSerial()) ++counters.forgetSinceInsert;
                }
                keepVariant = patchedCopy != nullptr ? patchedCopy : variant;
                cached = true;
                ++dispatchCacheHits;
                if (untouched) {
                    if (restamped != 0) variant->generation.store(restamped, std::memory_order_release);

                    if (rank != 0) {
                        auto rotated = std::make_shared<DispatchEntry>();
                        rotated->variants.reserve(variants.size());
                        rotated->variants.push_back(variant);
                        for (std::size_t i = 0; i < variants.size(); ++i) {
                            if (i != rank) rotated->variants.push_back(variants[i]);
                        }
                        rotated->touched = entry->touched;
                        rotated->order = entry->order;
                        again->second = std::move(rotated);
                    }

                    if (dispatchCacheHits - again->second->touched > dispatchCacheEntries() / 8) {
                        dispatchOrder.splice(dispatchOrder.begin(), dispatchOrder, again->second->order);
                        again->second->touched = dispatchCacheHits;
                        ++counters.touches;
                    }
                }
            } else {
                if (traceCache) std::fprintf(stderr, "[dispatch-cache] 0x%llx %s\n", static_cast<unsigned long long>(address), outcome == EntryOutcome::QueuedStores ? "queued stores over the captured memory" : "captured memory changed");

                missedEntry = entry;
                missedDiffering = outcome == EntryOutcome::Differing;
                if (!untouched) ++counters.replaced;
            }
            if (profile && std::chrono::steady_clock::now() - counters.lastReport > std::chrono::seconds(10)) {
                counters.lastReport = std::chrono::steady_clock::now();
                reportDispatchCache(counters);
            }
            phaseTiming.Phase(PhaseRelock);
        }
    }
}

}
