#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"

#ifdef _WIN32
#include <windows.h>
#include <synchapi.h>

namespace AgcDriver {

void WaitWhile(const std::atomic<std::uint32_t>& word, std::uint32_t value) noexcept {
    auto* address = const_cast<std::atomic<std::uint32_t>*>(&word);
    while (word.load(std::memory_order_relaxed) == value) WaitOnAddress(address, &value, sizeof(value), INFINITE);
}

void WakeOne(std::atomic<std::uint32_t>& word) noexcept {
    WakeByAddressSingle(&word);
}

void WakeAll(std::atomic<std::uint32_t>& word) noexcept {
    WakeByAddressAll(&word);
}

}
#endif
