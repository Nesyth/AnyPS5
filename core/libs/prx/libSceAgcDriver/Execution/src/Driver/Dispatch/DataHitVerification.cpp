#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace AgcDriver::DriverDetail {

void Driver::verifyDataHit(const ShaderSnapshot& snapshot, std::size_t codeOffset, ShaderRecompiler::RecompileRequest request, std::span<const ShaderRecompiler::MemoryRegion> memory, std::uint64_t address, const DispatchVariant& variant, std::span<const std::uint32_t> liveWords, const ShaderRecompiler::RecompileResult& patched) {
    const auto fail = [&](const char* what, std::size_t position, std::size_t slot) {
        std::fprintf(stderr, "[dispatch-cache] APS5_VERIFY_DATA_HITS: %s disagrees (program 0x%llx, position %zu, slot %zu)\n", what, static_cast<unsigned long long>(address), position, slot);
        std::fflush(stderr);
        std::abort();
    };
    auto verifyMemory = std::make_shared<ShaderMemory>(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter(), &QueryPendingSpan);
    std::shared_ptr<const PreparedShaderState::UserDataLink> linking;
    auto invocation = InvocationFor(snapshot, codeOffset, request, false, &linking);
    const auto capture = CaptureInvocation(*verifyMemory, snapshot, codeOffset, request, invocation, &linking);
    const auto regions = verifyMemory->Regions();
    request.context.memory = regions;
    const auto result = invocation.Materialize(*capture);
    if (!variant.runBases.empty()) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> held;
        for (const auto& run : variant.runs) {
            if (!held.empty() && held.back().second == run.first) held.back().second = run.second;
            else held.push_back(run);
        }
        std::vector<std::pair<std::uint64_t, std::uint64_t>> fresh;
        std::vector<std::byte> bytes;
        for (const auto& region : dispatchRegions(memory, regions, linking.get())) {
            if (!fresh.empty() && fresh.back().second == region.guestAddress) fresh.back().second += region.bytes.size();
            else fresh.emplace_back(region.guestAddress, region.guestAddress + region.bytes.size());
            bytes.insert(bytes.end(), region.bytes.begin(), region.bytes.end());
        }
        if (held != fresh) fail("the read set", 0, held.size());
        if (bytes.size() != liveWords.size() * sizeof(std::uint32_t) || std::memcmp(bytes.data(), liveWords.data(), bytes.size()) != 0) fail("the captured bytes", 0, 0);
    } else {
        std::size_t position = 0, run = 0;
        for (const auto& region : dispatchRegions(memory, regions, linking.get())) {
            if (run >= variant.runs.size() || variant.runs[run].first != region.guestAddress || variant.runs[run].second != region.guestAddress + region.bytes.size()) fail("the read set", position, run);
            const auto count = region.bytes.size() / sizeof(std::uint32_t);
            if (position + count > liveWords.size() || std::memcmp(region.bytes.data(), liveWords.data() + position, region.bytes.size()) != 0) fail("the captured bytes", position, run);
            position += count;
            ++run;
        }
        if (run != variant.runs.size()) fail("the read set", position, run);
    }
    if (result->variantId != patched.variantId) fail("the variant", 0, 0);
    if (result->bindings.size() != patched.bindings.size() || result->pushConstants != patched.pushConstants) fail("the bindings", 0, 0);
    for (std::size_t b = 0; b < result->bindings.size(); ++b) {
        const auto& fresh = result->bindings[b];
        const auto& reused = patched.bindings[b];
        if (fresh.kind != reused.kind || fresh.role != reused.role || fresh.binding != reused.binding || fresh.count != reused.count || fresh.guestDescriptor != reused.guestDescriptor) fail("the bindings", b, 0);
    }
    std::lock_guard cacheLock(dispatchCacheMutex);
    ++entryCounters.dataVerified;
}

void Driver::verifyRelocatedHit(const ShaderSnapshot& snapshot, std::size_t codeOffset, ShaderRecompiler::RecompileRequest request, std::span<const ShaderRecompiler::MemoryRegion> memory, std::uint64_t address, const DispatchVariant& variant, std::span<const std::uint32_t> liveWords, const ShaderRecompiler::RecompileResult& relocated) {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_DISPATCH_RELOCATE");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    static std::atomic<std::uint64_t> seen{0}, checked{0}, differing{0}, printed{0};
    if (every == 0 || seen.fetch_add(1, std::memory_order_relaxed) % every != 0) return;
    std::vector<std::string> differences;
    const auto note = [&](const char* what, std::size_t binding, std::size_t word, std::uint64_t relocatedValue, std::uint64_t freshValue) {
        char text[192];
        std::snprintf(text, sizeof(text), "%s differs at binding %zu word %zu: relocated 0x%llx, fresh 0x%llx", what, binding, word, static_cast<unsigned long long>(relocatedValue), static_cast<unsigned long long>(freshValue));
        differences.emplace_back(text);
    };
    try {
        auto verifyMemory = std::make_shared<ShaderMemory>(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter(), &QueryPendingSpan);
        auto invocation = InvocationFor(snapshot, codeOffset, request);
        const auto capture = CaptureInvocation(*verifyMemory, snapshot, codeOffset, request, invocation);
        const auto regions = verifyMemory->Regions();
        request.context.memory = regions;
        const auto fresh = invocation.Materialize(*capture);
        std::vector<std::pair<std::uint64_t, std::uint32_t>> held;
        std::size_t offset = 0;
        for (const auto& [begin, end] : variant.runs) {
            const auto count = static_cast<std::size_t>((end - begin) / sizeof(std::uint32_t));
            for (std::size_t word = 0; word < count && offset + word < liveWords.size(); ++word) held.emplace_back(begin + word * sizeof(std::uint32_t), liveWords[offset + word]);
            offset += count;
        }
        std::vector<std::pair<std::uint64_t, std::uint32_t>> read;
        for (const auto& region : regions) {
            const bool registered = std::any_of(memory.begin(), memory.end(), [&](const auto& known) { return region.guestAddress >= known.guestAddress && region.guestAddress < known.guestAddress + known.bytes.size(); });
            if (registered) continue;
            for (std::size_t at = 0; at + sizeof(std::uint32_t) <= region.bytes.size(); at += sizeof(std::uint32_t)) {
                std::uint32_t value = 0;
                std::memcpy(&value, region.bytes.data() + at, sizeof(value));
                read.emplace_back(region.guestAddress + at, value);
            }
        }
        std::sort(held.begin(), held.end());
        std::sort(read.begin(), read.end());
        if (held.size() != read.size()) note("the read set size", 0, 0, held.size(), read.size());
        for (std::size_t at = 0; at < held.size() && at < read.size(); ++at) {
            if (held[at].first != read[at].first) {
                note("the read set address", 0, at, held[at].first, read[at].first);
                break;
            }
            if (held[at].second != read[at].second) {
                note("the read bytes", 0, at, held[at].second, read[at].second);
                break;
            }
        }
        if (fresh->PipelineVariantId() != relocated.PipelineVariantId()) note("the variant", 0, 0, relocated.PipelineVariantId(), fresh->PipelineVariantId());
        if (fresh->bindings.size() != relocated.bindings.size()) note("the binding count", 0, 0, relocated.bindings.size(), fresh->bindings.size());
        for (std::size_t b = 0; b < fresh->bindings.size() && b < relocated.bindings.size(); ++b) {
            const auto& a = relocated.bindings[b];
            const auto& f = fresh->bindings[b];
            if (a.kind != f.kind || a.role != f.role || a.binding != f.binding || a.count != f.count) note("the binding shape", b, 0, static_cast<std::uint64_t>(a.role), static_cast<std::uint64_t>(f.role));
            if (a.guestDescriptor.size() != f.guestDescriptor.size()) {
                note("the descriptor size", b, 0, a.guestDescriptor.size(), f.guestDescriptor.size());
                continue;
            }
            for (std::size_t word = 0; word < f.guestDescriptor.size(); ++word) {
                if (a.guestDescriptor[word] == f.guestDescriptor[word]) continue;
                note("the guest descriptor", b, word, a.guestDescriptor[word], f.guestDescriptor[word]);
                break;
            }
        }
        if (fresh->pushConstants.size() != relocated.pushConstants.size()) note("the push constant size", 0, 0, relocated.pushConstants.size(), fresh->pushConstants.size());
        for (std::size_t at = 0; at + sizeof(std::uint32_t) <= fresh->pushConstants.size() && at + sizeof(std::uint32_t) <= relocated.pushConstants.size(); at += sizeof(std::uint32_t)) {
            std::uint32_t a = 0, f = 0;
            std::memcpy(&a, relocated.pushConstants.data() + at, sizeof(a));
            std::memcpy(&f, fresh->pushConstants.data() + at, sizeof(f));
            if (a == f) continue;
            note("the push constants", 0, at / sizeof(std::uint32_t), a, f);
            break;
        }
        if (fresh->specialization.size() != relocated.specialization.size()) note("the specialization count", 0, 0, relocated.specialization.size(), fresh->specialization.size());
        for (std::size_t at = 0; at < fresh->specialization.size() && at < relocated.specialization.size(); ++at) {
            if (fresh->specialization[at].id == relocated.specialization[at].id && fresh->specialization[at].value == relocated.specialization[at].value) continue;
            note("the specialization", 0, at, relocated.specialization[at].value, fresh->specialization[at].value);
            break;
        }
    } catch (const std::exception& error) {
        differences.emplace_back(std::string("the fresh capture threw: ") + error.what());
    }
    const auto count = checked.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!differences.empty()) differing.fetch_add(1, std::memory_order_relaxed);
    for (const auto& difference : differences) {
        if (printed.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[dispatch-relocate-verify] program 0x%llx: %s\n", static_cast<unsigned long long>(address), difference.c_str());
    }
    if (count % 4096 == 0) std::fprintf(stderr, "[dispatch-relocate-verify] %llu checked, %llu differing\n", static_cast<unsigned long long>(count), static_cast<unsigned long long>(differing.load(std::memory_order_relaxed)));
}

}
