#include "prx/libSceAgcDriver/Execution/include/WorkerSampler.hpp"
#include "prx/libSceAgcDriver/Execution/include/LockProbe.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include "prx/libc/include/HostThread.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef _WIN32
#include "prx/libSceAgcDriver/Execution/include/StackWalk.hpp"
#include <windows.h>
#include <tlhelp32.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <stop_token>
#include <system_error>
#include <thread>
#include <vector>
#endif

namespace AgcDriver {

#ifdef _WIN32
bool ReadableStackRegion(std::uint64_t address, std::uint64_t& low, std::uint64_t& high) {
    constexpr DWORD readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    MEMORY_BASIC_INFORMATION memory{};
    if (VirtualQuery(reinterpret_cast<LPCVOID>(address), &memory, sizeof(memory)) == 0 || memory.State != MEM_COMMIT) return false;
    if ((memory.Protect & (PAGE_GUARD | PAGE_NOACCESS)) != 0 || (memory.Protect & readable) == 0) return false;
    low = reinterpret_cast<std::uint64_t>(memory.BaseAddress);
    high = low + memory.RegionSize;
    return true;
}

namespace {

std::jthread processSamplerThread;
std::jthread workerSamplerThread;
std::jthread committerSamplerThread;
std::jthread finisherSamplerThread;

std::optional<std::string> ConfiguredPath(const wchar_t* value) {
    if (value == nullptr) return std::nullopt;
    const auto text = std::filesystem::path(value).u8string();
    return std::string(reinterpret_cast<const char*>(text.data()), text.size());
}

std::filesystem::path NativePath(const std::string& path) {
    return std::filesystem::path(reinterpret_cast<const char8_t*>(path.c_str()));
}

std::FILE* OpenForWriting(const std::string& path) {
    return _wfopen(NativePath(path).c_str(), L"w");
}

double HitchMs() {
    static const double configured = [] {
        const char* text = std::getenv("APS5_SAMPLE_HITCH_MS");
        return text != nullptr ? std::strtod(text, nullptr) : 0.0;
    }();
    return configured;
}

double HitchAfterMs() {
    static const double configured = [] {
        const char* text = std::getenv("APS5_SAMPLE_HITCH_AFTER_S");
        return text != nullptr ? std::strtod(text, nullptr) * 1e3 : 0.0;
    }();
    return configured;
}

struct Sampler;
struct ProcessSampler;
std::atomic<ProcessSampler*> hitchProcess{nullptr};
constexpr std::size_t HitchThreads = 3;
std::array<std::atomic<DWORD>, HitchThreads> hitchThreadIds{};
std::array<std::atomic<Sampler*>, HitchThreads> hitchSamplers{};

// Debug aid: APS5_SAMPLE_WORKER=<file> samples the calling thread every millisecond and, every 20 s,
// writes "module+offset self inclusive" lines to <file> for offline symbolization with nm.
struct Sampler {
    static constexpr int ProcessDepth = 12;
    static constexpr int WorkerDepth = 64;

    HANDLE target = nullptr;
    std::string path;
    int depth = ProcessDepth;
    bool keepStacks = false;
    std::uint64_t stackLow = 0;
    std::uint64_t stackHigh = 0;
    std::vector<DwarfFrames> modules;
    std::uint64_t pendingModule = 0;
    AgcDriver::Mutex mutex;
    std::map<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>> counts;
    CollapsedStacks stacks;
    std::uint64_t samples = 0;
    std::int64_t suspendedTicks = 0;
    std::int64_t longestSuspendedTicks = 0;
    static constexpr int KeptDepth = 48;
    static constexpr std::uint64_t HitchLimit = 400;
    struct Kept {
        std::int64_t ticks = 0;
        int count = 0;
        std::array<std::uint64_t, KeptDepth> frames{};
    };
    std::vector<Kept> ring;
    std::size_t ringNext = 0;
    std::string hitchPath;
    std::int64_t startedTicks = 0;
    std::uint64_t hitches = 0;

    ~Sampler() noexcept(false) {
        if (target != nullptr && !CloseHandle(target)) throw std::system_error(GetLastError(), std::system_category(), "Closing sampler thread handle");
    }

    bool readStack(std::uint64_t address, std::uint64_t& value) const {
        if (stackHigh < stackLow + sizeof(value) || address < stackLow || address > stackHigh - sizeof(value)) return false;
        std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
        return true;
    }

    const DwarfFrames* moduleOf(std::uint64_t pc) const {
        const auto found = std::upper_bound(modules.begin(), modules.end(), pc, [](std::uint64_t value, const DwarfFrames& module) { return value < module.Base(); });
        return found != modules.begin() && std::prev(found)->Contains(pc) ? &*std::prev(found) : nullptr;
    }

    bool hasUnwindData(std::uint64_t pc) const {
        DWORD64 imageBase = 0;
        const auto* module = moduleOf(pc - 1);
        return RtlLookupFunctionEntry(pc, &imageBase, nullptr) != nullptr || (module != nullptr && module->Covers(pc - 1));
    }

    DwarfFrames::Step stepDwarf(CONTEXT& context, bool leaf) {
        const auto* module = moduleOf(context.Rip - (leaf ? 0 : 1));
        if (module == nullptr) {
            MEMORY_BASIC_INFORMATION memory{};
            if (VirtualQuery(reinterpret_cast<LPCVOID>(context.Rip), &memory, sizeof(memory)) != 0 && memory.Type == MEM_IMAGE) pendingModule = reinterpret_cast<std::uint64_t>(memory.AllocationBase);
            return DwarfFrames::Step::Failed;
        }
        DwarfFrames::Registers registers{context.Rax, context.Rdx, context.Rcx, context.Rbx, context.Rsi, context.Rdi, context.Rbp, context.Rsp, context.R8, context.R9, context.R10, context.R11, context.R12, context.R13, context.R14, context.R15, context.Rip};
        const auto step = module->Unwind(registers, leaf, [&](std::uint64_t address, std::uint64_t& value) { return readStack(address, value); });
        if (step != DwarfFrames::Step::Stepped) return step;
        DWORD64* columns[] = {&context.Rax, &context.Rdx, &context.Rcx, &context.Rbx, &context.Rsi, &context.Rdi, &context.Rbp, &context.Rsp, &context.R8, &context.R9, &context.R10, &context.R11, &context.R12, &context.R13, &context.R14, &context.R15, &context.Rip};
        for (std::size_t column = 0; column < registers.size(); ++column) *columns[column] = registers[column];
        return step;
    }

    int walk(CONTEXT& context, std::array<std::uint64_t, WorkerDepth>& frames) {
        if (context.Rsp < stackLow || context.Rsp >= stackHigh) {
            stackLow = 0;
            stackHigh = 0;
            ReadableStackRegion(context.Rsp, stackLow, stackHigh);
        }
        const bool dwarf = stackHigh != 0 && context.Rsp >= stackLow && context.Rsp < stackHigh;
        int count = 0;
        while (count < depth && context.Rip != 0) {
            frames[count++] = context.Rip;
            DWORD64 imageBase = 0;
            if (auto* entry = RtlLookupFunctionEntry(context.Rip, &imageBase, nullptr)) {
                PVOID handler = nullptr;
                DWORD64 establisher = 0;
                RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, context.Rip, entry, &context, &handler, &establisher, nullptr);
                continue;
            }
            if (!dwarf) break;
            const auto step = stepDwarf(context, count == 1);
            if (step == DwarfFrames::Step::Stepped) continue;
            std::uint64_t returnAddress = 0;
            if (step == DwarfFrames::Step::Failed || count != 1 || !readStack(context.Rsp, returnAddress) || !hasUnwindData(returnAddress)) break;
            context.Rip = returnAddress;
            context.Rsp += sizeof(returnAddress);
        }
        return count;
    }

    void indexModule() {
        const auto base = std::exchange(pendingModule, 0);
        if (moduleOf(base) != nullptr) return;
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
        const auto* sections = IMAGE_FIRST_SECTION(nt);
        const DwarfFrames::Byte* frames = nullptr;
        const DwarfFrames::Byte* framesEnd = nullptr;
        for (unsigned index = 0; index < nt->FileHeader.NumberOfSections; ++index) {
            if (std::memcmp(sections[index].Name, ".ehfram", 8) != 0) continue;
            frames = reinterpret_cast<const DwarfFrames::Byte*>(base + sections[index].VirtualAddress);
            framesEnd = frames + sections[index].Misc.VirtualSize;
        }
        const auto at = std::upper_bound(modules.begin(), modules.end(), base, [](std::uint64_t value, const DwarfFrames& module) { return value < module.Base(); });
        modules.emplace(at, base, base + nt->OptionalHeader.SizeOfImage, frames, framesEnd);
    }

    void sample() {
        CONTEXT context{};
        context.ContextFlags = CONTEXT_FULL;
        std::array<std::uint64_t, WorkerDepth> frames;
        int count = 0;
        LARGE_INTEGER suspended{};
        LARGE_INTEGER resumed{};
        QueryPerformanceCounter(&suspended);
        if (SuspendThread(target) == static_cast<DWORD>(-1)) return;
        if (GetThreadContext(target, &context) != 0) count = walk(context, frames);
        ResumeThread(target);
        QueryPerformanceCounter(&resumed);
        if (pendingModule != 0) indexModule();
        std::lock_guard lock(mutex);
        ++samples;
        suspendedTicks += resumed.QuadPart - suspended.QuadPart;
        longestSuspendedTicks = std::max(longestSuspendedTicks, resumed.QuadPart - suspended.QuadPart);
        for (int i = 0; i < count; ++i) {
            auto& entry = counts[frames[i]];
            if (i == 0) ++entry.first;
            ++entry.second;
        }
        if (keepStacks && count != 0) stacks.Add({frames.data(), static_cast<std::size_t>(count)}, context.Rip != 0);
        if (!ring.empty()) {
            auto& kept = ring[ringNext++ % ring.size()];
            kept.ticks = suspended.QuadPart;
            kept.count = std::min(count, KeptDepth);
            std::copy_n(frames.begin(), kept.count, kept.frames.begin());
        }
    }

    void writeHitch(std::int64_t from, std::int64_t to, const char* label, std::uint32_t value);

    std::size_t ringCount(std::int64_t from, std::int64_t to) {
        std::lock_guard lock(mutex);
        std::size_t inside = 0;
        for (const auto& kept : ring) inside += kept.count != 0 && kept.ticks >= from && kept.ticks <= to;
        return inside;
    }

    void writeRing(std::FILE* file, std::int64_t from, std::int64_t to, double tickMs) {
        std::lock_guard lock(mutex);
        std::map<std::uint64_t, std::string> loaded;
        for (std::size_t step = 0; step < ring.size(); ++step) {
            const auto& kept = ring[(ringNext + step) % ring.size()];
            if (kept.count == 0 || kept.ticks < from || kept.ticks > to) continue;
            std::fprintf(file, "  %.2f", static_cast<double>(kept.ticks - from) * tickMs);
            for (int i = 0; i < kept.count; ++i) std::fprintf(file, "%c%s", i == 0 ? ' ' : ';', frameName(kept.frames[i], loaded).c_str());
            std::fputc('\n', file);
        }
    }

    static std::string frameName(std::uint64_t address, std::map<std::uint64_t, std::string>& loaded) {
        HMODULE module = nullptr;
        char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(address), &module)) GetModuleFileNameA(module, name, sizeof(name));
        const char* base = std::strrchr(name, '\\');
        if (module != nullptr) loaded.emplace(reinterpret_cast<std::uint64_t>(module), base ? base + 1 : name);
        char frame[MAX_PATH + 32];
        std::snprintf(frame, sizeof(frame), "%s+0x%llx", base ? base + 1 : name, static_cast<unsigned long long>(address - reinterpret_cast<std::uint64_t>(module)));
        return frame;
    }

    static void writeModules(std::FILE* file, const std::map<std::uint64_t, std::string>& loaded) {
        for (const auto& [base, name] : loaded) {
            const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
            const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
            std::fprintf(file, "# module %s 0x%lx 0x%lx\n", name.c_str(), static_cast<unsigned long>(nt->OptionalHeader.SizeOfImage), static_cast<unsigned long>(nt->FileHeader.TimeDateStamp));
        }
    }

    void write() {
        std::lock_guard lock(mutex);
        std::FILE* file = OpenForWriting(path);
        if (file == nullptr) return;
        std::fprintf(file, "# %llu samples\n", static_cast<unsigned long long>(samples));
        std::map<std::uint64_t, std::string> loaded;
        for (const auto& [address, count] : counts) {
            HMODULE module = nullptr;
            char name[MAX_PATH] = "?";
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(address), &module)) GetModuleFileNameA(module, name, sizeof(name));
            const char* base = std::strrchr(name, '\\');
            if (module != nullptr) loaded.emplace(reinterpret_cast<std::uint64_t>(module), base ? base + 1 : name);
            std::fprintf(file, "%s 0x%llx %llu %llu\n", base ? base + 1 : name, static_cast<unsigned long long>(address - reinterpret_cast<std::uint64_t>(module)), static_cast<unsigned long long>(count.first), static_cast<unsigned long long>(count.second));
        }
        writeModules(file, loaded);
        std::fclose(file);
        if (keepStacks) writeStacks();
    }

    void writeStacks() {
        const auto dot = path.rfind('.');
        const auto slash = path.find_last_of("\\/");
        const auto stem = dot != std::string::npos && (slash == std::string::npos || dot > slash) ? path.substr(0, dot) : path;
        const auto collapsed = stem + ".collapsed.txt";
        const auto temporary = collapsed + ".tmp";
        std::map<std::uint64_t, std::string> names;
        std::map<std::uint64_t, std::string> loaded;
        for (const auto& [address, count] : counts) names.emplace(address, frameName(address, loaded));
        std::FILE* file = OpenForWriting(temporary);
        if (file == nullptr) return;
        LARGE_INTEGER frequency{};
        QueryPerformanceFrequency(&frequency);
        const double tickUs = 1e6 / static_cast<double>(frequency.QuadPart);
        std::fprintf(file, "# samples=%llu depth=%d truncated=%llu suspended_us_mean=%.1f suspended_us_max=%.1f\n", static_cast<unsigned long long>(samples), depth, static_cast<unsigned long long>(stacks.Truncated()),
            samples != 0 ? static_cast<double>(suspendedTicks) * tickUs / static_cast<double>(samples) : 0.0, static_cast<double>(longestSuspendedTicks) * tickUs);
        writeModules(file, loaded);
        const auto text = stacks.Format([&](std::uint64_t address) { return names.at(address); });
        const bool written = std::fwrite(text.data(), 1, text.size(), file) == text.size();
        if (std::fclose(file) != 0 || !written) return;
        MoveFileExW(NativePath(temporary).c_str(), NativePath(collapsed).c_str(), MOVEFILE_REPLACE_EXISTING);
    }
};

void Sampler::writeHitch(std::int64_t from, std::int64_t to, const char* label, std::uint32_t value) {
    std::lock_guard lock(mutex);
    LARGE_INTEGER frequency{};
    QueryPerformanceFrequency(&frequency);
    const double tickMs = 1e3 / static_cast<double>(frequency.QuadPart);
    if (hitches >= HitchLimit || static_cast<double>(from - startedTicks) * tickMs < HitchAfterMs()) return;
    std::FILE* file = _wfopen(NativePath(hitchPath).c_str(), hitches == 0 ? L"w" : L"a");
    if (file == nullptr) return;
    ++hitches;
    std::size_t inside = 0;
    for (const auto& kept : ring) inside += kept.count != 0 && kept.ticks >= from && kept.ticks <= to;
    std::fprintf(file, "hitch %llu %s 0x%x %.2f ms at %.1f ms, %zu samples\n", static_cast<unsigned long long>(hitches), label, value, static_cast<double>(to - from) * tickMs, static_cast<double>(from - startedTicks) * tickMs, inside);
    std::map<std::uint64_t, std::string> loaded;
    for (std::size_t step = 0; step < ring.size(); ++step) {
        const auto& kept = ring[(ringNext + step) % ring.size()];
        if (kept.count == 0 || kept.ticks < from || kept.ticks > to) continue;
        std::fprintf(file, "  %.2f", static_cast<double>(kept.ticks - from) * tickMs);
        for (int i = 0; i < kept.count; ++i) std::fprintf(file, "%c%s", i == 0 ? ' ' : ';', frameName(kept.frames[i], loaded).c_str());
        std::fputc('\n', file);
    }
    std::fclose(file);
}

// Debug aid: APS5_SAMPLE_THREADS=<file> samples every thread in the process every 2 ms and, every
// 20 s, writes each busy thread's hottest frames ("module+offset self inclusive") to <file>.
struct ProcessSampler {
    static constexpr int HitchDepth = 24;
    static constexpr std::size_t HitchRing = 256;

    std::string path;
    DWORD self = 0;
    std::map<DWORD, Sampler> threads;
    std::uint64_t rounds = 0;
    AgcDriver::Mutex mapMutex;
    std::string hitchPath;
    std::int64_t startedTicks = 0;
    std::uint64_t hitches = 0;

    void writeHitch(DWORD caller, std::int64_t from, std::int64_t to, const char* label, std::uint32_t value) {
        std::lock_guard lock(mapMutex);
        LARGE_INTEGER frequency{};
        QueryPerformanceFrequency(&frequency);
        const double tickMs = 1e3 / static_cast<double>(frequency.QuadPart);
        if (hitches >= Sampler::HitchLimit || static_cast<double>(from - startedTicks) * tickMs < HitchAfterMs()) return;
        std::FILE* file = _wfopen(NativePath(hitchPath).c_str(), hitches == 0 ? L"w" : L"a");
        if (file == nullptr) return;
        ++hitches;
        std::fprintf(file, "hitch %llu %s 0x%x %.2f ms at %.1f ms, caller %lu\n", static_cast<unsigned long long>(hitches), label, value, static_cast<double>(to - from) * tickMs, static_cast<double>(from - startedTicks) * tickMs, static_cast<unsigned long>(caller));
        for (auto& [id, thread] : threads) {
            if (thread.target == nullptr || thread.ringCount(from, to) == 0) continue;
            std::string name;
            PWSTR description = nullptr;
            if (SUCCEEDED(GetThreadDescription(thread.target, &description)) && description != nullptr) {
                const int bytes = WideCharToMultiByte(CP_UTF8, 0, description, -1, nullptr, 0, nullptr, nullptr);
                if (bytes > 1) {
                    name.resize(static_cast<std::size_t>(bytes - 1));
                    WideCharToMultiByte(CP_UTF8, 0, description, -1, name.data(), bytes, nullptr, nullptr);
                }
                LocalFree(description);
            }
            std::fprintf(file, " thread %lu %s\n", static_cast<unsigned long>(id), name.c_str());
            thread.writeRing(file, from, to, tickMs);
        }
        std::fclose(file);
    }

    std::chrono::steady_clock::time_point enumerated{};

    // The system-wide thread snapshot is slow (tens of ms); the thread list is refreshed every 5 s and
    // the rounds in between only sample the known threads.
    void enumerate() {
        const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (snapshot == INVALID_HANDLE_VALUE) return;
        THREADENTRY32 entry{};
        entry.dwSize = sizeof(entry);
        const DWORD process = GetCurrentProcessId();
        std::lock_guard lock(mapMutex);
        for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
            if (entry.th32OwnerProcessID != process || entry.th32ThreadID == self) continue;
            auto& thread = threads[entry.th32ThreadID];
            if (thread.target == nullptr) thread.target = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, entry.th32ThreadID);
            if (HitchMs() > 0.0 && thread.ring.empty()) {
                thread.ring.resize(HitchRing);
                thread.depth = HitchDepth;
            }
        }
        CloseHandle(snapshot);
        enumerated = std::chrono::steady_clock::now();
    }

    void sample() {
        if (threads.empty() || std::chrono::steady_clock::now() - enumerated > std::chrono::seconds(5)) enumerate();
        for (auto& [id, thread] : threads) {
            if (thread.target != nullptr) thread.sample();
        }
        ++rounds;
    }

    void write() {
        std::FILE* file = OpenForWriting(path);
        if (file == nullptr) return;
        std::fprintf(file, "# %llu rounds\n", static_cast<unsigned long long>(rounds));
        std::map<std::uint64_t, std::string> loaded;
        for (auto& [id, thread] : threads) {
            std::lock_guard lock(thread.mutex);
            std::vector<std::pair<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>>> hot(thread.counts.begin(), thread.counts.end());
            std::sort(hot.begin(), hot.end(), [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
            std::uint64_t leaves = 0;
            for (const auto& [address, count] : hot) leaves += count.first;
            if (leaves == 0) continue;
            std::string name;
            PWSTR description = nullptr;
            if (SUCCEEDED(GetThreadDescription(thread.target, &description)) && description != nullptr) {
                const int bytes = WideCharToMultiByte(CP_UTF8, 0, description, -1, nullptr, 0, nullptr, nullptr);
                if (bytes > 1) {
                    name.resize(static_cast<std::size_t>(bytes - 1));
                    WideCharToMultiByte(CP_UTF8, 0, description, -1, name.data(), bytes, nullptr, nullptr);
                }
                LocalFree(description);
            }
            std::fprintf(file, "thread %lu samples %llu name %s\n", static_cast<unsigned long>(id), static_cast<unsigned long long>(thread.samples), name.c_str());
            for (std::size_t i = 0; i < hot.size() && i < 40; ++i) {
                const auto address = hot[i].first;
                HMODULE module = nullptr;
                char name[MAX_PATH] = "?";
                if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(address), &module)) GetModuleFileNameA(module, name, sizeof(name));
                const char* base = std::strrchr(name, '\\');
                if (module != nullptr) loaded.emplace(reinterpret_cast<std::uint64_t>(module), base ? base + 1 : name);
                std::fprintf(file, "  %s 0x%llx %llu %llu\n", base ? base + 1 : name, static_cast<unsigned long long>(address - reinterpret_cast<std::uint64_t>(module)), static_cast<unsigned long long>(hot[i].second.first), static_cast<unsigned long long>(hot[i].second.second));
            }
        }
        Sampler::writeModules(file, loaded);
        std::fclose(file);
    }
};

void StartProcessSampler() {
    const auto path = ConfiguredPath(_wgetenv(L"APS5_SAMPLE_THREADS"));
    if (!path) return;
    processSamplerThread = Libc::StartHostJThread([path = *path](std::stop_token token) {
        auto sampler = std::make_unique<ProcessSampler>();
        sampler->path = path;
        sampler->self = GetCurrentThreadId();
        if (HitchMs() > 0.0) {
            const auto dot = path.rfind('.');
            const auto slash = path.find_last_of("\\/");
            sampler->hitchPath = (dot != std::string::npos && (slash == std::string::npos || dot > slash) ? path.substr(0, dot) : path) + ".hitches.txt";
            LARGE_INTEGER now{};
            QueryPerformanceCounter(&now);
            sampler->startedTicks = now.QuadPart;
            hitchProcess.store(sampler.get(), std::memory_order_release);
        }
        auto flushed = std::chrono::steady_clock::now();
        while (!token.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            sampler->sample();
            if (std::chrono::steady_clock::now() - flushed > std::chrono::seconds(20)) {
                sampler->write();
                flushed = std::chrono::steady_clock::now();
            }
        }
        sampler->write();
    });
}

void StartCallingThreadSampler(const std::optional<std::string>& path, std::jthread& slot) {
    if (!path) return;
    auto sampler = std::make_unique<Sampler>();
    sampler->path = *path;
    sampler->depth = Sampler::WorkerDepth;
    sampler->keepStacks = true;
    const auto* block = reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
    MEMORY_BASIC_INFORMATION stack{};
    if (VirtualQuery(block->StackLimit, &stack, sizeof(stack)) != 0) {
        sampler->stackLow = reinterpret_cast<std::uint64_t>(stack.AllocationBase);
        sampler->stackHigh = reinterpret_cast<std::uint64_t>(block->StackBase);
    }
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &sampler->target, THREAD_ALL_ACCESS, FALSE, 0)) throw std::system_error(GetLastError(), std::system_category(), "Duplicating sampler thread handle");
    if (HitchMs() > 0.0) {
        const auto dot = path->rfind('.');
        const auto slash = path->find_last_of("\\/");
        sampler->hitchPath = (dot != std::string::npos && (slash == std::string::npos || dot > slash) ? path->substr(0, dot) : *path) + ".hitches.txt";
        sampler->ring.resize(1024);
        LARGE_INTEGER now{};
        QueryPerformanceCounter(&now);
        sampler->startedTicks = now.QuadPart;
        for (std::size_t index = 0; index < HitchThreads; ++index) {
            DWORD free = 0;
            if (!hitchThreadIds[index].compare_exchange_strong(free, GetCurrentThreadId())) continue;
            hitchSamplers[index].store(sampler.get(), std::memory_order_release);
            break;
        }
    }
    slot = Libc::StartHostJThread([sampler = std::move(sampler)](std::stop_token token) {
        auto flushed = std::chrono::steady_clock::now();
        while (!token.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            sampler->sample();
            if (std::chrono::steady_clock::now() - flushed > std::chrono::seconds(20)) {
                sampler->write();
                flushed = std::chrono::steady_clock::now();
            }
        }
        sampler->write();
    });
}

}

void StartWorkerSampler(std::uint32_t queue) {
    if (queue == 0) StartProcessSampler();
    static const auto sampledQueue = [] {
        const char* value = std::getenv("APS5_SAMPLE_WORKER_QUEUE");
        return value != nullptr ? static_cast<std::uint32_t>(std::strtoul(value, nullptr, 0)) : 0u;
    }();
    if (queue == sampledQueue) StartCallingThreadSampler(ConfiguredPath(_wgetenv(L"APS5_SAMPLE_WORKER")), workerSamplerThread);
}

void StartCommitterSampler() {
    StartCallingThreadSampler(ConfiguredPath(_wgetenv(L"APS5_SAMPLE_COMMITTER")), committerSamplerThread);
}

void StartFinisherSampler() {
    StartCallingThreadSampler(ConfiguredPath(_wgetenv(L"APS5_SAMPLE_FINISHER")), finisherSamplerThread);
}

SampledHitch::SampledHitch(const char* label, std::uint32_t value, bool armed) noexcept : label(label), value(value), began(0) {
    if (!armed || HitchMs() <= 0.0) return;
    LARGE_INTEGER now{};
    QueryPerformanceCounter(&now);
    began = now.QuadPart;
}

SampledHitch::~SampledHitch() {
    if (began == 0) return;
    LARGE_INTEGER now{};
    LARGE_INTEGER frequency{};
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&frequency);
    if (static_cast<double>(now.QuadPart - began) * 1e3 / static_cast<double>(frequency.QuadPart) < HitchMs()) return;
    const DWORD self = GetCurrentThreadId();
    for (std::size_t index = 0; index < HitchThreads; ++index) {
        if (hitchThreadIds[index].load(std::memory_order_relaxed) != self) continue;
        if (auto* sampler = hitchSamplers[index].load(std::memory_order_acquire)) sampler->writeHitch(began, now.QuadPart, label, value);
        break;
    }
    if (auto* process = hitchProcess.load(std::memory_order_acquire)) process->writeHitch(self, began, now.QuadPart, label, value);
}

double LockProbeMs() noexcept {
    static const double configured = [] {
        const char* text = std::getenv("APS5_SAMPLE_LOCK_WAIT_MS");
        return text != nullptr ? std::strtod(text, nullptr) : 0.0;
    }();
    return configured;
}

std::uint32_t LockProbeThread() noexcept {
    return GetCurrentThreadId();
}

void ReportLockWait(const char* lock, const void* address, std::uint32_t owner, std::uint32_t reader, double waitedMs) {
    static const auto path = ConfiguredPath(_wgetenv(L"APS5_SAMPLE_LOCK_WAITS"));
    static AgcDriver::Mutex mutex;
    static std::uint64_t reports = 0;
    if (!path) return;
    std::lock_guard guard(mutex);
    if (reports >= 300u) return;
    std::FILE* file = _wfopen(NativePath(*path).c_str(), reports == 0 ? L"w" : L"a");
    if (file == nullptr) return;
    ++reports;
    const auto nameOf = [](HANDLE thread) {
        std::string name;
        PWSTR description = nullptr;
        if (SUCCEEDED(GetThreadDescription(thread, &description)) && description != nullptr) {
            const int bytes = WideCharToMultiByte(CP_UTF8, 0, description, -1, nullptr, 0, nullptr, nullptr);
            if (bytes > 1) {
                name.resize(static_cast<std::size_t>(bytes - 1));
                WideCharToMultiByte(CP_UTF8, 0, description, -1, name.data(), bytes, nullptr, nullptr);
            }
            LocalFree(description);
        }
        return name;
    };
    const DWORD self = GetCurrentThreadId();
    std::fprintf(file, "wait %llu on %s (%p): thread %lu '%s' has waited %.1f ms; last exclusive owner %lu, last shared owner %lu\n", static_cast<unsigned long long>(reports), lock, address, static_cast<unsigned long>(self), nameOf(GetCurrentThread()).c_str(), waitedMs, static_cast<unsigned long>(owner), static_cast<unsigned long>(reader));
    std::map<std::uint64_t, std::string> loaded;
    const std::array<DWORD, 2> holders{static_cast<DWORD>(owner), reader != owner ? static_cast<DWORD>(reader) : 0u};
    for (const DWORD id : holders) {
        if (id == 0 || id == self) continue;
        Sampler probe;
        probe.target = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, id);
        if (probe.target == nullptr) {
            std::fprintf(file, " thread %lu: gone\n", static_cast<unsigned long>(id));
            continue;
        }
        probe.depth = Sampler::KeptDepth;
        probe.ring.resize(1);
        probe.sample();
        const auto& kept = probe.ring[0];
        std::fprintf(file, " thread %lu '%s' (%s):", static_cast<unsigned long>(id), nameOf(probe.target).c_str(), id == owner ? "exclusive" : "shared");
        for (int i = 0; i < kept.count; ++i) std::fprintf(file, "%c%s", i == 0 ? ' ' : ';', Sampler::frameName(kept.frames[i], loaded).c_str());
        std::fputc('\n', file);
    }
    std::fclose(file);
}

void StopWorkerSampler() {
    hitchProcess.store(nullptr, std::memory_order_release);
    for (auto& sampler : hitchSamplers) sampler.store(nullptr, std::memory_order_release);
    processSamplerThread.request_stop();
    workerSamplerThread.request_stop();
    committerSamplerThread.request_stop();
    finisherSamplerThread.request_stop();
    if (processSamplerThread.joinable()) processSamplerThread.join();
    if (workerSamplerThread.joinable()) workerSamplerThread.join();
    if (committerSamplerThread.joinable()) committerSamplerThread.join();
    if (finisherSamplerThread.joinable()) finisherSamplerThread.join();
}
#else
bool ReadableStackRegion(std::uint64_t, std::uint64_t&, std::uint64_t&) { return false; }
double LockProbeMs() noexcept {
    return 0.0;
}

std::uint32_t LockProbeThread() noexcept {
    return 0;
}

void ReportLockWait(const char*, const void*, std::uint32_t, std::uint32_t, double) {}

SampledHitch::SampledHitch(const char* label, std::uint32_t value, bool) noexcept : label(label), value(value), began(0) {}
SampledHitch::~SampledHitch() {}
void StartWorkerSampler(std::uint32_t) {}
void StartCommitterSampler() {}
void StartFinisherSampler() {}
void StopWorkerSampler() {}
#endif

}
