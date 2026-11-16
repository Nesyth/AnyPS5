#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_LOCKPROBE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_LOCKPROBE_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

namespace AgcDriver {

double LockProbeMs() noexcept;
std::uint32_t LockProbeThread() noexcept;
void ReportLockWait(const char* lock, const void* address, std::uint32_t owner, std::uint32_t reader, double waitedMs);

template<typename TAttempt>
void AwaitProbedLock(const char* lock, const void* address, const std::atomic<std::uint32_t>& owner, const std::atomic<std::uint32_t>& reader, TAttempt attempt) {
    const auto began = std::chrono::steady_clock::now();
    bool reported = false;
    for (unsigned turns = 0; !attempt(); ++turns) {
        if (turns < 128u) continue;
        std::this_thread::yield();
        if (reported) continue;
        const double waited = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count();
        if (waited < LockProbeMs()) continue;
        reported = true;
        ReportLockWait(lock, address, owner.load(std::memory_order_relaxed), reader.load(std::memory_order_relaxed), waited);
    }
}

template<typename TMutex>
class ProbedSharedMutex {
public:
    explicit ProbedSharedMutex(const char* name) : name(name) {
        inner.lock();
        inner.unlock();
    }
    ProbedSharedMutex(const ProbedSharedMutex&) = delete;
    ProbedSharedMutex& operator=(const ProbedSharedMutex&) = delete;

    void lock() {
        if (LockProbeMs() <= 0.0) {
            inner.lock();
            return;
        }
        if (!inner.try_lock()) AwaitProbedLock(name, this, owner, reader, [&] { return inner.try_lock(); });
        owner.store(LockProbeThread(), std::memory_order_relaxed);
    }

    bool try_lock() {
        if (!inner.try_lock()) return false;
        if (LockProbeMs() > 0.0) owner.store(LockProbeThread(), std::memory_order_relaxed);
        return true;
    }

    void unlock() {
        inner.unlock();
    }

    void lock_shared() {
        if (LockProbeMs() <= 0.0) {
            inner.lock_shared();
            return;
        }
        if (!inner.try_lock_shared()) AwaitProbedLock(name, this, owner, reader, [&] { return inner.try_lock_shared(); });
        reader.store(LockProbeThread(), std::memory_order_relaxed);
    }

    bool try_lock_shared() {
        if (!inner.try_lock_shared()) return false;
        if (LockProbeMs() > 0.0) reader.store(LockProbeThread(), std::memory_order_relaxed);
        return true;
    }

    void unlock_shared() {
        inner.unlock_shared();
    }

private:
    TMutex inner;
    const char* name;
    std::atomic<std::uint32_t> owner{0};
    std::atomic<std::uint32_t> reader{0};
};

}

#endif
