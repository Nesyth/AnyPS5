#include "prx/libSceAgcDriver/Execution/include/ThreadPriority.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#include <sys/qos.h>
#else
#include <dlfcn.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace AgcDriver {

namespace {

ThreadPriorityMode RequestedMode() {
    static const ThreadPriorityMode mode = ParseThreadPriorityMode(std::getenv("APS5_THREAD_PRIORITY"));
    return mode;
}

int EnvInt(const char* name, int fallback) {
    const char* text = std::getenv(name);
    return text != nullptr && *text != '\0' ? std::atoi(text) : fallback;
}

void LogOnce(const char* role, const std::string& text) {
    static std::mutex mutex;
    static std::set<std::string> seen;
    std::lock_guard lock(mutex);
    if (!seen.insert(std::string(role) + "|" + text).second) return;
    std::fprintf(stderr, "[thread-priority] %s: %s\n", role, text.c_str());
}

#if defined(__APPLE__)

void Raise(const char* role) {
    if (RequestedMode() == ThreadPriorityMode::Realtime) {
        mach_timebase_info_data_t timebase{};
        mach_timebase_info(&timebase);
        const auto ticks = [&](std::uint64_t nanos) {
            return static_cast<std::uint32_t>(nanos * timebase.denom / timebase.numer);
        };
        thread_time_constraint_policy_data_t policy{ticks(1000000), ticks(500000), ticks(1000000), TRUE};
        const auto status = thread_policy_set(pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
                                              reinterpret_cast<thread_policy_t>(&policy), THREAD_TIME_CONSTRAINT_POLICY_COUNT);
        if (status != KERN_SUCCESS)
            throw std::runtime_error(std::string("APS5_THREAD_PRIORITY=rt: ") + role + " time-constraint policy refused (kern_return " + std::to_string(status) + ")");
        LogOnce(role, "Mach time-constraint policy");
        return;
    }
    const int error = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    if (error != 0)
        throw std::runtime_error(std::string("APS5_THREAD_PRIORITY=high: ") + role + " user-interactive QoS refused: " + std::strerror(error));
    LogOnce(role, "user-interactive QoS");
}

#elif !defined(_WIN32)

struct DBusConnection;
struct DBusMessage;
struct DBusError {
    const char* name;
    const char* message;
    unsigned int dummy1 : 1, dummy2 : 1, dummy3 : 1, dummy4 : 1, dummy5 : 1;
    void* padding1;
};
constexpr int DbusBusSystem = 1;
constexpr int DbusTypeInvalid = 0;
constexpr int DbusTypeInt32 = static_cast<int>('i');
constexpr int DbusTypeUint32 = static_cast<int>('u');
constexpr int DbusTypeUint64 = static_cast<int>('t');
constexpr int DbusTypeInt64 = static_cast<int>('x');
constexpr int DbusTypeString = static_cast<int>('s');
constexpr int DbusTypeVariant = static_cast<int>('v');

struct DBusMessageIter {
    void* dummy1;
    void* dummy2;
    std::uint32_t dummy3;
    int dummy4, dummy5, dummy6, dummy7, dummy8, dummy9, dummy10, dummy11;
    int pad1;
    void* pad2;
    void* pad3;
};

struct Dbus {
    void* handle = nullptr;
    DBusConnection* connection = nullptr;
    int (*threadsInitDefault)() = nullptr;
    void (*errorInit)(DBusError*) = nullptr;
    void (*errorFree)(DBusError*) = nullptr;
    int (*errorIsSet)(const DBusError*) = nullptr;
    DBusConnection* (*busGet)(int, DBusError*) = nullptr;
    DBusMessage* (*messageNewMethodCall)(const char*, const char*, const char*, const char*) = nullptr;
    int (*messageAppendArgs)(DBusMessage*, int, ...) = nullptr;
    DBusMessage* (*sendWithReplyAndBlock)(DBusConnection*, DBusMessage*, int, DBusError*) = nullptr;
    void (*messageUnref)(DBusMessage*) = nullptr;
    int (*iterInit)(DBusMessage*, DBusMessageIter*) = nullptr;
    int (*iterGetArgType)(DBusMessageIter*) = nullptr;
    void (*iterRecurse)(DBusMessageIter*, DBusMessageIter*) = nullptr;
    void (*iterGetBasic)(DBusMessageIter*, void*) = nullptr;
    std::string failure;
};

Dbus& LoadDbus() {
    static Dbus dbus = [] {
        Dbus loaded;
        loaded.handle = dlopen("libdbus-1.so.3", RTLD_NOW | RTLD_LOCAL);
        if (loaded.handle == nullptr) {
            loaded.failure = "libdbus-1.so.3 is not available";
            return loaded;
        }
        const auto resolve = [&](auto& function, const char* name) {
            function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(dlsym(loaded.handle, name));
            if (function == nullptr && loaded.failure.empty()) loaded.failure = std::string("libdbus lacks ") + name;
        };
        resolve(loaded.threadsInitDefault, "dbus_threads_init_default");
        resolve(loaded.errorInit, "dbus_error_init");
        resolve(loaded.errorFree, "dbus_error_free");
        resolve(loaded.errorIsSet, "dbus_error_is_set");
        resolve(loaded.busGet, "dbus_bus_get");
        resolve(loaded.messageNewMethodCall, "dbus_message_new_method_call");
        resolve(loaded.messageAppendArgs, "dbus_message_append_args");
        resolve(loaded.sendWithReplyAndBlock, "dbus_connection_send_with_reply_and_block");
        resolve(loaded.messageUnref, "dbus_message_unref");
        resolve(loaded.iterInit, "dbus_message_iter_init");
        resolve(loaded.iterGetArgType, "dbus_message_iter_get_arg_type");
        resolve(loaded.iterRecurse, "dbus_message_iter_recurse");
        resolve(loaded.iterGetBasic, "dbus_message_iter_get_basic");
        if (!loaded.failure.empty()) return loaded;
        loaded.threadsInitDefault();
        DBusError error{};
        loaded.errorInit(&error);
        loaded.connection = loaded.busGet(DbusBusSystem, &error);
        if (loaded.connection == nullptr) {
            loaded.failure = std::string("system bus: ") + (loaded.errorIsSet(&error) && error.message != nullptr ? error.message : "unavailable");
            loaded.errorFree(&error);
        }
        return loaded;
    }();
    return dbus;
}

DBusMessage* RtkitSend(DBusMessage* message, const char* method, std::string& failure) {
    auto& dbus = LoadDbus();
    DBusError error{};
    dbus.errorInit(&error);
    auto* reply = dbus.sendWithReplyAndBlock(dbus.connection, message, 5000, &error);
    dbus.messageUnref(message);
    if (reply == nullptr) {
        failure = std::string(method) + ": " + (dbus.errorIsSet(&error) && error.message != nullptr ? error.message : "no reply");
        dbus.errorFree(&error);
    }
    return reply;
}

bool RtkitCall(const char* method, std::uint64_t tid, int type, const void* value, std::string& failure) {
    auto& dbus = LoadDbus();
    if (dbus.connection == nullptr) {
        failure = dbus.failure;
        return false;
    }
    static std::mutex mutex;
    std::lock_guard lock(mutex);
    auto* message = dbus.messageNewMethodCall("org.freedesktop.RealtimeKit1", "/org/freedesktop/RealtimeKit1", "org.freedesktop.RealtimeKit1", method);
    if (message == nullptr) {
        failure = "dbus message allocation failed";
        return false;
    }
    const std::uint64_t pid = static_cast<std::uint64_t>(getpid());
    bool ok = false;
    if (type == DbusTypeInt32) ok = dbus.messageAppendArgs(message, DbusTypeUint64, &pid, DbusTypeUint64, &tid, DbusTypeInt32, value, DbusTypeInvalid) != 0;
    else ok = dbus.messageAppendArgs(message, DbusTypeUint64, &pid, DbusTypeUint64, &tid, DbusTypeUint32, value, DbusTypeInvalid) != 0;
    if (!ok) {
        dbus.messageUnref(message);
        failure = "dbus argument append failed";
        return false;
    }
    auto* reply = RtkitSend(message, method, failure);
    if (reply == nullptr) return false;
    dbus.messageUnref(reply);
    return true;
}

bool RtkitProperty(const char* name, std::int64_t& value, std::string& failure) {
    auto& dbus = LoadDbus();
    if (dbus.connection == nullptr) {
        failure = dbus.failure;
        return false;
    }
    auto* message = dbus.messageNewMethodCall("org.freedesktop.RealtimeKit1", "/org/freedesktop/RealtimeKit1", "org.freedesktop.DBus.Properties", "Get");
    if (message == nullptr) {
        failure = "dbus message allocation failed";
        return false;
    }
    const char* interfaceName = "org.freedesktop.RealtimeKit1";
    if (dbus.messageAppendArgs(message, DbusTypeString, &interfaceName, DbusTypeString, &name, DbusTypeInvalid) == 0) {
        dbus.messageUnref(message);
        failure = "dbus argument append failed";
        return false;
    }
    auto* reply = RtkitSend(message, name, failure);
    if (reply == nullptr) return false;
    DBusMessageIter iter{};
    DBusMessageIter variant{};
    bool ok = false;
    if (dbus.iterInit(reply, &iter) != 0 && dbus.iterGetArgType(&iter) == DbusTypeVariant) {
        dbus.iterRecurse(&iter, &variant);
        const int type = dbus.iterGetArgType(&variant);
        if (type == DbusTypeInt32) {
            std::int32_t number = 0;
            dbus.iterGetBasic(&variant, &number);
            value = number;
            ok = true;
        } else if (type == DbusTypeInt64) {
            std::int64_t number = 0;
            dbus.iterGetBasic(&variant, &number);
            value = number;
            ok = true;
        }
    }
    dbus.messageUnref(reply);
    if (!ok) failure = std::string(name) + ": unexpected reply type";
    return ok;
}

bool PrepareRealtime(std::uint32_t priority, const char* role, std::string& failure) {
    std::int64_t maxPriority = 0;
    std::int64_t maxRttime = 0;
    std::string rtkitFailure;
    std::optional<RtkitLimits> rtkit;
    if (RtkitProperty("MaxRealtimePriority", maxPriority, rtkitFailure) && RtkitProperty("RTTimeUSecMax", maxRttime, rtkitFailure)) rtkit = RtkitLimits{maxPriority, maxRttime};
    static std::mutex mutex;
    std::lock_guard lock(mutex);
    rlimit limit{};
    if (getrlimit(RLIMIT_RTTIME, &limit) != 0) {
        failure = "getrlimit(RLIMIT_RTTIME) failed";
        return false;
    }
    const auto requested = static_cast<std::uint64_t>(std::max(1, EnvInt("APS5_THREAD_RTTIME_US", 200000)));
    const auto plan = PlanRealtime(priority, rtkit, {limit.rlim_cur, limit.rlim_max}, requested);
    if (!plan.refusal.empty()) {
        failure = rtkit.has_value() ? plan.refusal : rtkitFailure;
        return false;
    }
    sched_param param{};
    if (sched_getparam(0, &param) != 0 || sched_setscheduler(0, sched_getscheduler(0) | SCHED_RESET_ON_FORK, &param) != 0) {
        failure = "SCHED_RESET_ON_FORK could not be set";
        return false;
    }
    if (!plan.limit.has_value()) return true;
    const rlimit wanted{plan.limit->soft, plan.limit->hard};
    if (setrlimit(RLIMIT_RTTIME, &wanted) != 0) {
        failure = "setrlimit(RLIMIT_RTTIME) failed";
        return false;
    }
    if (wanted.rlim_max != limit.rlim_max) {
        const auto text = [](rlim_t value) { return value == RLIM_INFINITY ? std::string("unlimited") : std::to_string(value); };
        LogOnce(role, "RLIMIT_RTTIME hard limit lowered from " + text(limit.rlim_max) + " to " + text(wanted.rlim_max) + " us for the process");
    }
    return true;
}

void Raise(const char* role) {
    const auto tid = static_cast<std::uint64_t>(syscall(SYS_gettid));
    std::string failure;
    if (RequestedMode() == ThreadPriorityMode::Realtime) {
        const std::uint32_t priority = static_cast<std::uint32_t>(std::max(1, std::min(99, EnvInt("APS5_THREAD_RT", 10))));
        if (!PrepareRealtime(priority, role, failure) || !RtkitCall("MakeThreadRealtimeWithPID", tid, DbusTypeUint32, &priority, failure)) {
            throw std::runtime_error(std::string("APS5_THREAD_PRIORITY=rt: ") + role + " realtime refused: " + failure);
        }
        const int policy = sched_getscheduler(0) & ~SCHED_RESET_ON_FORK;
        LogOnce(role, std::string(policy == SCHED_FIFO ? "SCHED_FIFO " : "SCHED_RR ") + std::to_string(priority) + " via rtkit");
        return;
    }
    const std::int32_t nice = static_cast<std::int32_t>(std::max(-20, std::min(19, EnvInt("APS5_THREAD_NICE", -10))));
    if (!RtkitCall("MakeThreadHighPriorityWithPID", tid, DbusTypeInt32, &nice, failure)) {
        throw std::runtime_error(std::string("APS5_THREAD_PRIORITY=high: ") + role + " nice " + std::to_string(nice) + " refused: " + failure);
    }
    LogOnce(role, "nice " + std::to_string(nice) + " via rtkit");
}

#else

void Raise(const char* role) {
    const int priority = RequestedMode() == ThreadPriorityMode::Realtime ? THREAD_PRIORITY_TIME_CRITICAL : THREAD_PRIORITY_HIGHEST;
    if (!SetThreadPriority(GetCurrentThread(), priority)) {
        throw std::runtime_error(std::string("APS5_THREAD_PRIORITY: ") + role + " SetThreadPriority failed (error " + std::to_string(GetLastError()) + ")");
    }
    LogOnce(role, priority == THREAD_PRIORITY_TIME_CRITICAL ? "time-critical" : "highest");
}

#endif

}

ThreadPriorityMode ParseThreadPriorityMode(const char* text) {
    if (text == nullptr || *text == '\0' || std::strcmp(text, "off") == 0) return ThreadPriorityMode::Off;
    if (std::strcmp(text, "high") == 0) return ThreadPriorityMode::High;
    if (std::strcmp(text, "rt") == 0) return ThreadPriorityMode::Realtime;
    throw std::invalid_argument(std::string("APS5_THREAD_PRIORITY=") + text + " is not off, high or rt");
}

RealtimePlan PlanRealtime(std::uint32_t priority, const std::optional<RtkitLimits>& rtkit, RttimeLimit current, std::uint64_t requestedUs) {
    if (!rtkit.has_value()) return {"rtkit's limits are unknown", std::nullopt};
    if (static_cast<std::int64_t>(priority) > rtkit->maxPriority) return {"priority " + std::to_string(priority) + " is above rtkit's MaxRealtimePriority " + std::to_string(rtkit->maxPriority), std::nullopt};
    if (rtkit->maxRttimeUs <= 0) return {"rtkit reports RTTimeUSecMax " + std::to_string(rtkit->maxRttimeUs), std::nullopt};
    const auto ceiling = std::min(requestedUs, static_cast<std::uint64_t>(rtkit->maxRttimeUs));
    auto wanted = current;
    if (wanted.hard > ceiling) wanted.hard = ceiling;
    if (wanted.soft > wanted.hard / 2) wanted.soft = wanted.hard / 2;
    if (wanted == current) return {};
    return {{}, wanted};
}

void RaiseWorkerThreadPriority(const char* role) {
    if (RequestedMode() == ThreadPriorityMode::Off) return;
    Raise(role);
}

#ifdef _WIN32
namespace {

// Cores reserved for driver threads (CoreIndex values) and the threads that took them: a later
// reservation keeps the earlier ones out of the process default and leaves their threads where they are.
std::mutex reservationMutex;
std::set<unsigned> reservedCores;
std::set<DWORD> reservedThreads;

void reserveCore(const char* role, const char* cpuVariable) {
    const char* text = cpuVariable != nullptr ? std::getenv(cpuVariable) : nullptr;
    using GetInformation = BOOL(WINAPI*)(void*, ULONG, PULONG, HANDLE, ULONG);
    using SetCpuSets = BOOL(WINAPI*)(HANDLE, const ULONG*, ULONG);
    const auto kernel = GetModuleHandleW(L"kernel32.dll");
    const auto getInformation = reinterpret_cast<GetInformation>(reinterpret_cast<void*>(GetProcAddress(kernel, "GetSystemCpuSetInformation")));
    const auto setDefault = reinterpret_cast<SetCpuSets>(reinterpret_cast<void*>(GetProcAddress(kernel, "SetProcessDefaultCpuSets")));
    const auto setSelected = reinterpret_cast<SetCpuSets>(reinterpret_cast<void*>(GetProcAddress(kernel, "SetThreadSelectedCpuSets")));
    if (getInformation == nullptr || setDefault == nullptr || setSelected == nullptr) return;
    std::lock_guard lock(reservationMutex);
    ULONG bytes = 0;
    getInformation(nullptr, 0, &bytes, GetCurrentProcess(), 0);
    std::vector<unsigned char> buffer(bytes);
    if (bytes == 0 || !getInformation(buffer.data(), bytes, &bytes, GetCurrentProcess(), 0)) return;
    // SYSTEM_CPU_SET_INFORMATION: Size at 0, Type at 4 (0 = CpuSetInformation), Id at 8,
    // LogicalProcessorIndex at 14, CoreIndex at 15.
    const auto byteAt = [&](ULONG at) { return static_cast<unsigned>(buffer[at]); };
    const auto wordAt = [&](ULONG at) { ULONG value = 0; std::memcpy(&value, buffer.data() + at, sizeof(value)); return value; };
    // Without the variable: the first logical processor of the highest-numbered physical core not reserved
    // yet, on a host with at least six cores (fewer would leave the title's spinning job workers too little room).
    std::set<unsigned> cores;
    for (ULONG offset = 0; offset + 16 <= bytes && wordAt(offset) != 0; offset += wordAt(offset)) {
        if (wordAt(offset + 4) == 0) cores.insert(byteAt(offset + 15));
    }
    if (cores.empty() || ((text == nullptr || *text == 0) && cores.size() < 6)) return;
    unsigned wanted = 0x100;
    if (text != nullptr && *text != 0) {
        wanted = static_cast<unsigned>(std::strtoul(text, nullptr, 10));
    } else {
        unsigned pick = 0x100;
        for (auto it = cores.rbegin(); it != cores.rend() && pick == 0x100; ++it) {
            if (!reservedCores.contains(*it)) pick = *it;
        }
        if (pick == 0x100) return;
        for (ULONG offset = 0; offset + 16 <= bytes && wordAt(offset) != 0; offset += wordAt(offset)) {
            if (wordAt(offset + 4) == 0 && byteAt(offset + 15) == pick) wanted = std::min(wanted, byteAt(offset + 14));
        }
    }
    unsigned core = 0x100;
    ULONG own = 0;
    for (ULONG offset = 0; offset + 16 <= bytes && wordAt(offset) != 0; offset += wordAt(offset)) {
        if (wordAt(offset + 4) == 0 && byteAt(offset + 14) == wanted) {
            core = byteAt(offset + 15);
            own = wordAt(offset + 8);
        }
    }
    if (core == 0x100) return;
    reservedCores.insert(core);
    reservedThreads.insert(GetCurrentThreadId());
    std::vector<ULONG> others;
    for (ULONG offset = 0; offset + 16 <= bytes && wordAt(offset) != 0; offset += wordAt(offset)) {
        if (wordAt(offset + 4) == 0 && !reservedCores.contains(byteAt(offset + 15))) others.push_back(wordAt(offset + 8));
    }
    const bool process = setDefault(GetCurrentProcess(), others.data(), static_cast<ULONG>(others.size())) != 0;
    const bool thread = setSelected(GetCurrentThread(), &own, 1) != 0;
    // Threads already running (the title's spinning job workers among them) may stay on the reserved core
    // until the scheduler moves them, so they select the other CPU sets explicitly; later threads take the
    // process default. A thread pinned with an affinity mask keeps its mask.
    std::size_t moved = 0;
    if (const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0); snapshot != INVALID_HANDLE_VALUE) {
        THREADENTRY32 entry{};
        entry.dwSize = sizeof(entry);
        const DWORD processId = GetCurrentProcessId();
        for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
            if (entry.th32OwnerProcessID != processId || reservedThreads.contains(entry.th32ThreadID)) continue;
            if (const HANDLE other = OpenThread(THREAD_SET_LIMITED_INFORMATION, FALSE, entry.th32ThreadID); other != nullptr) {
                if (setSelected(other, others.data(), static_cast<ULONG>(others.size()))) ++moved;
                CloseHandle(other);
            }
        }
        CloseHandle(snapshot);
    }
    std::fprintf(stderr, "[placement] %s on logical %u (core %u reserved, %zu reserved in all: process default %s, thread %s, %zu running threads moved; %zu other CPU sets)\n", role, wanted, core, reservedCores.size(), process ? "set" : "failed", thread ? "set" : "failed", moved, others.size());
}

}
#endif

void PlaceReservedThread(const char* role) {
#ifdef _WIN32
    if (std::getenv("APS5_NO_COMMITTER_CORE") != nullptr) return;
    reserveCore(role, "APS5_COMMITTER_CPU");
#else
    (void)role;
#endif
}

void PlaceFinisherThread(const char* role) {
#ifdef _WIN32
    if (std::getenv("APS5_XP_FINISHER_CORE") == nullptr) return;
    reserveCore(role, nullptr);
#else
    (void)role;
#endif
}

}
