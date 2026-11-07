#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_HOSTTHREADLOCAL_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_HOSTTHREADLOCAL_HPP

#include <memory>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#include "prx/libc/include/HostThreadSlot.hpp"
#endif

#ifdef _WIN32
template<typename TTag>
struct HostThreadLocalCache {};
#endif

template<typename TValue, typename TTag>
TValue& HostThreadLocal() {
#ifdef _WIN32
    using Cache = HostThreadSlot<TValue*, HostThreadLocalCache<TTag>>;
    if (auto* cached = Cache::Get(); cached != nullptr) return *cached;
    static const DWORD slot = FlsAlloc(+[](void* value) {
        Cache::Set(nullptr);
        delete static_cast<TValue*>(value);
    });
    if (slot == FLS_OUT_OF_INDEXES) throw std::runtime_error("host thread-local slot allocation failed");
    auto* value = static_cast<TValue*>(FlsGetValue(slot));
    if (value == nullptr) {
        auto created = std::make_unique<TValue>();
        if (!FlsSetValue(slot, created.get())) throw std::runtime_error("host thread-local value registration failed");
        value = created.release();
    }
    Cache::Set(value);
    return *value;
#else
    static thread_local TValue value;
    return value;
#endif
}

#endif
