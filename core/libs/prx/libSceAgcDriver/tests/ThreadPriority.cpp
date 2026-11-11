#include "prx/libSceAgcDriver/Execution/include/ThreadPriority.hpp"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <algorithm>
#include <cstring>
#include <exception>
#include <thread>
#include <vector>
#endif

namespace {

using namespace AgcDriver;

constexpr std::uint64_t Unlimited = std::numeric_limits<std::uint64_t>::max();
constexpr RtkitLimits Rtkit{20, 200000};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void ParseTests() {
    for (const char* text : {static_cast<const char*>(nullptr), "", "off"}) Require(ParseThreadPriorityMode(text) == ThreadPriorityMode::Off, std::string("APS5_THREAD_PRIORITY=") + (text == nullptr ? "(unset)" : text) + " must leave priorities alone");
    Require(ParseThreadPriorityMode("rt") == ThreadPriorityMode::Realtime, "APS5_THREAD_PRIORITY=rt must ask for real-time priority");
    Require(ParseThreadPriorityMode("high") == ThreadPriorityMode::High, "APS5_THREAD_PRIORITY=high must ask for a nice level");
    for (const char* text : {"hgih", "HIGH", "realtime", "on", "1", "0", " high"}) {
        bool threw = false;
        try {
            static_cast<void>(ParseThreadPriorityMode(text));
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        Require(threw, std::string("APS5_THREAD_PRIORITY=") + text + " must be rejected");
    }
}

void ExpectRefused(const RealtimePlan& plan, const std::string& what) {
    Require(!plan.refusal.empty() && !plan.limit.has_value(), what + " must refuse real-time priority without touching RLIMIT_RTTIME");
}

void ExpectLimit(const RealtimePlan& plan, RttimeLimit expected, const std::string& what) {
    Require(plan.refusal.empty() && plan.limit.has_value(), what + " must set RLIMIT_RTTIME");
    Require(*plan.limit == expected, what + " set RLIMIT_RTTIME to soft " + std::to_string(plan.limit->soft) + " hard " + std::to_string(plan.limit->hard) + ", expected soft " + std::to_string(expected.soft) + " hard " + std::to_string(expected.hard));
}

void PlanTests() {
    ExpectRefused(PlanRealtime(10, std::nullopt, {Unlimited, Unlimited}, 200000), "an unreachable rtkit");
    ExpectRefused(PlanRealtime(10, std::nullopt, {30000, 150000}, 200000), "an unreachable rtkit");
    ExpectRefused(PlanRealtime(21, Rtkit, {Unlimited, Unlimited}, 200000), "a priority above rtkit's MaxRealtimePriority");
    ExpectRefused(PlanRealtime(10, RtkitLimits{20, 0}, {Unlimited, Unlimited}, 200000), "rtkit without an RTTimeUSecMax");
    ExpectLimit(PlanRealtime(20, Rtkit, {Unlimited, Unlimited}, 200000), {100000, 200000}, "an unlimited hard limit");
    ExpectLimit(PlanRealtime(10, Rtkit, {100000, Unlimited}, 200000), {100000, 200000}, "a lowered soft limit under an unlimited hard limit");
    ExpectLimit(PlanRealtime(10, Rtkit, {Unlimited, Unlimited}, 500000), {100000, 200000}, "APS5_THREAD_RTTIME_US above RTTimeUSecMax");
    ExpectLimit(PlanRealtime(10, Rtkit, {Unlimited, Unlimited}, 100000), {50000, 100000}, "APS5_THREAD_RTTIME_US below RTTimeUSecMax");
    ExpectLimit(PlanRealtime(10, Rtkit, {150000, 150000}, 200000), {75000, 150000}, "a hard limit rtkit accepts with a soft limit above half of it");
    const auto kept = PlanRealtime(10, Rtkit, {30000, 150000}, 200000);
    Require(kept.refusal.empty() && !kept.limit.has_value(), "limits rtkit already accepts must be left alone");
}

void RefusalTest() {
    bool threw = false;
    try {
        RaiseWorkerThreadPriority("test");
    } catch (const std::runtime_error& error) {
        threw = true;
        std::cout << "refused: " << error.what() << '\n';
    }
    Require(threw, "a refused priority request must throw");
}

#ifdef _WIN32
template <typename Function>
Function Kernel32(const char* name) {
    return reinterpret_cast<Function>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), name)));
}

std::vector<ULONG> ProcessDefaultCpuSets() {
    const auto get = Kernel32<BOOL(WINAPI*)(HANDLE, PULONG, ULONG, PULONG)>("GetProcessDefaultCpuSets");
    Require(get != nullptr, "GetProcessDefaultCpuSets is missing");
    ULONG count = 0;
    get(GetCurrentProcess(), nullptr, 0, &count);
    std::vector<ULONG> sets(count);
    Require(count == 0 || get(GetCurrentProcess(), sets.data(), count, &count), "GetProcessDefaultCpuSets failed");
    sets.resize(count);
    std::sort(sets.begin(), sets.end());
    return sets;
}

std::size_t CpuSetCount() {
    const auto get = Kernel32<BOOL(WINAPI*)(void*, ULONG, PULONG, HANDLE, ULONG)>("GetSystemCpuSetInformation");
    Require(get != nullptr, "GetSystemCpuSetInformation is missing");
    ULONG bytes = 0;
    get(nullptr, 0, &bytes, GetCurrentProcess(), 0);
    std::vector<unsigned char> buffer(bytes);
    Require(bytes != 0 && get(buffer.data(), bytes, &bytes, GetCurrentProcess(), 0), "GetSystemCpuSetInformation failed");
    std::size_t count = 0;
    for (ULONG offset = 0; offset + 8 <= bytes;) {
        ULONG size = 0;
        ULONG type = 0;
        std::memcpy(&size, buffer.data() + offset, sizeof(size));
        std::memcpy(&type, buffer.data() + offset + 4, sizeof(type));
        if (size == 0) break;
        if (type == 0) ++count;
        offset += size;
    }
    return count;
}

// Queue worker 0's core: a reservation takes one core's CPU sets out of the process default, a second one
// by the same thread changes nothing, and a release puts every CPU set back. On a thread of its own, so
// the main thread keeps its CPU sets.
void PlacementTests() {
    std::exception_ptr failure;
    std::thread([&] {
        try {
            const auto all = CpuSetCount();
            PlaceQueueWorkerThread("test worker", true);
            const auto reserved = ProcessDefaultCpuSets();
            if (reserved.empty()) {
                std::cout << "no core reserved on this host (" << all << " CPU sets)\n";
                return;
            }
            Require(reserved.size() < all, "a reserved core did not leave the process default");
            PlaceQueueWorkerThread("test worker", true);
            Require(ProcessDefaultCpuSets() == reserved, "a second reservation by the same thread changed the process default");
            PlaceQueueWorkerThread("test worker", false);
            Require(ProcessDefaultCpuSets().size() == all, "a released core did not return to the process default");
            PlaceQueueWorkerThread("test worker", false);
            Require(ProcessDefaultCpuSets().size() == all, "a second release changed the process default");
        } catch (...) {
            failure = std::current_exception();
        }
    }).join();
    if (failure) std::rethrow_exception(failure);
}
#endif

}

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "refused") {
            RefusalTest();
            return 0;
        }
        ParseTests();
        PlanTests();
#ifdef _WIN32
        PlacementTests();
#endif
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
