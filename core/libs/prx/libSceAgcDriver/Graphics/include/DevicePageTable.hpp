#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEVICEPAGETABLE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEVICEPAGETABLE_HPP

#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

class DevicePageTable {
public:
    using Range = ShaderRecompiler::BdaAbi::Range;
    using Key = std::pair<std::uint64_t, std::uint64_t>;
    using Table = std::shared_ptr<const BdaPageTable>;
    struct Counts {
        std::uint64_t built = 0;
        std::uint64_t served = 0;
        std::uint64_t missed = 0;
        std::uint64_t superseded = 0;
        std::uint64_t refused = 0;
        std::uint64_t buildNanoseconds = 0;
    };
    static constexpr std::size_t KeptTables = 8;

    DevicePageTable() = default;
    ~DevicePageTable();
    DevicePageTable(const DevicePageTable&) = delete;
    DevicePageTable& operator=(const DevicePageTable&) = delete;
    Table Find(const Context& context, Key key, std::span<const Range> ranges);
    static std::vector<Range> Overlay(std::span<const Range> space, std::span<const Range> build);
    void WaitIdle();
    void Release(VkDevice device);
    Counts Totals() const;

private:
    struct Job {
        Context context;
        Key key;
        std::vector<Range> ranges;
    };
    struct Entry {
        VkDevice device = VK_NULL_HANDLE;
        Key key;
        Table table;
    };
    void work();
    mutable std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    std::optional<Job> pending;
    std::optional<Job> running;
    std::vector<Entry> entries;
    Counts counts;
    std::thread worker;
    bool stopping = false;
};

DevicePageTable& GuestPageTable();
void ReleaseGuestPageTable(VkDevice device);

}

#endif
