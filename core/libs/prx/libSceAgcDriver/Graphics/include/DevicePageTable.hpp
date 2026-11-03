#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEVICEPAGETABLE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEVICEPAGETABLE_HPP

#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace AgcDriver::Graphics {

class DevicePageTable {
public:
    using Range = ShaderRecompiler::BdaAbi::Range;
    enum class Cover : std::uint8_t { Covered, Missing, OtherAddress, Narrower };
    struct Snapshot {
        std::shared_ptr<const BdaPageTable> table;
        std::shared_ptr<const std::vector<Range>> ranges;
        std::shared_ptr<Buffer> header;
    };
    struct Counts {
        std::uint64_t builds = 0;
        std::uint64_t reuses = 0;
        std::uint64_t ranges = 0;
        std::uint64_t uploads = 0;
        std::uint64_t deferred = 0;
        std::uint64_t spaceBuilds = 0;
        std::uint64_t spaceReuses = 0;
        std::uint64_t refused = 0;
    };

    Snapshot Current(const Context& context);
    Snapshot ForRanges(const Context& context, std::uint64_t serial, std::uint64_t guard, std::span<const Range> ranges);
    void Release(VkDevice device);
    Counts Totals() const;
    static Cover Compare(std::span<const Range> table, std::span<const Range> build);

private:
    mutable std::mutex mutex;
    HostImportsProof proof;
    Snapshot snapshot;
    Counts counts;
    std::chrono::steady_clock::time_point built{};
    struct Space {
        VkDevice device = VK_NULL_HANDLE;
        std::uint64_t serial = 0;
        std::uint64_t guard = 0;
        Snapshot snapshot;
    };
    std::vector<Space> spaces;
    Snapshot finish(const Context& context, std::shared_ptr<const BdaPageTable> table, std::vector<Range> ranges);
};

DevicePageTable& GuestPageTable();
void ReleaseGuestPageTable(VkDevice device);

}

#endif
