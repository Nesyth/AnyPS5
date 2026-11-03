#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEVICEPAGETABLE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEVICEPAGETABLE_HPP

#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
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
    };
    struct Counts {
        std::uint64_t builds = 0;
        std::uint64_t reuses = 0;
        std::uint64_t ranges = 0;
    };

    Snapshot Current(const Context& context);
    void Release(VkDevice device);
    Counts Totals() const;
    static Cover Compare(std::span<const Range> table, std::span<const Range> build);

private:
    mutable std::mutex mutex;
    HostImportsProof proof;
    Snapshot snapshot;
    Counts counts;
};

DevicePageTable& GuestPageTable();
void ReleaseGuestPageTable(VkDevice device);

}

#endif
