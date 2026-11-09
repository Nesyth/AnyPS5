#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWRECORDS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWRECORDS_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRecordReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libc/include/HostThreadSlot.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

struct LeanShape;

struct DrawRecordKey {
    std::uint64_t shape = 0;
    std::uint64_t indexAddress = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t indexSize = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t flags = 0;
    std::uint32_t firstVertex = 0;
    std::uint32_t firstInstance = 0;
    bool indexed = false;

    static DrawRecordKey For(std::uint64_t shape, const Pm4::DrawParameters& parameters);
    std::uint64_t Probe() const;
    bool operator==(const DrawRecordKey&) const = default;
};

using DrawRecordHash = std::array<std::uint64_t, 2>;

DrawRecordHash HashDrawRecordWords(std::span<const std::uint32_t> first, std::span<const std::uint32_t> second);

struct DrawRecordSink {
    bool queued = false;
    std::uint64_t shape = 0;
    std::vector<std::uint32_t> contentKey;
    std::vector<std::uint32_t> vertexLayout;
    std::vector<std::uint32_t> vertexInputs;
    std::optional<std::uint32_t> fragmentOutputs;
    std::vector<DrawPipeline::Range> writes;

    void Reset();
    void NoteVertices(std::span<const std::optional<ShaderRecompiler::ShaderVertexStageInfo>> infos);
    DrawRecordHash Content() const { return HashDrawRecordWords(contentKey, vertexLayout); }
    DrawRecordHash Inputs() const { return HashDrawRecordWords(vertexInputs, {}); }
};

struct DrawRecordSinkTag {};
using DrawRecordSinkSlot = HostThreadSlot<DrawRecordSink*, DrawRecordSinkTag>;

struct DrawRecord : std::enable_shared_from_this<DrawRecord> {
    DrawRecordKey key;
    std::uint64_t probe = 0;
    std::shared_ptr<const LeanShape> shape;
    std::uint64_t deviceSerial = 0;
    std::uint64_t generation = 0;
    std::uint64_t builtPresent = 0;
    DrawRecordHash content{};
    DrawRecordHash inputs{};
    std::vector<std::uint32_t> contentKey;
    std::vector<std::uint32_t> vertexLayout;
    std::vector<std::uint32_t> vertexInputs;
    std::optional<std::uint32_t> fragmentOutputs;
    std::vector<DrawPipeline::Range> writes;

    static std::shared_ptr<DrawRecord> Build(const DrawRecordKey& key, std::shared_ptr<const LeanShape> shape, std::uint64_t deviceSerial, std::uint64_t generation, std::uint64_t present, const DrawRecordSink& sink);
    std::size_t Bytes() const;
};

struct DrawRecordSet {
    std::vector<std::shared_ptr<const DrawRecord>> records;
    std::size_t bytes = 0;

    const LeanShape* Shape() const;
    const DrawRecord* Match(const DrawRecordHash& content) const;
};

struct DrawRecordAnswer {
    std::shared_ptr<const DrawRecord> built;
    DrawRecordKey key;
    bool demote = false;
};

namespace DrawRecordFlags {

constexpr std::uint8_t On = 1;
constexpr std::uint8_t Probed = 2;
constexpr std::uint8_t BuildWanted = 4;
constexpr std::uint8_t VerifyDue = 8;

}

class DrawRecordStore {
public:
    static constexpr std::size_t Ways = 4;
    static constexpr std::size_t Sets = 2048;
    static constexpr std::size_t Variants = 4;
    static constexpr std::uint64_t KeepPresents = 3;
    static constexpr std::size_t DefaultByteCap = std::size_t{32} << 20;

    struct Entry {
        DrawRecordKey key;
        std::uint64_t probe = 0;
        std::uint64_t lastUse = 0;
        std::uint64_t lastPresent = 0;
        std::uint64_t lastSeq = 0;
        std::uint64_t buildSeq = 0;
        std::shared_ptr<const DrawRecordSet> records;
        bool used = false;
        bool refused = false;

        bool BuildPending(std::uint64_t front) const { return buildSeq != 0 && buildSeq >= front; }
    };

    enum class Installed : std::uint8_t { Yes, NoEntry, Refused, OverCap };

    Entry* Find(const DrawRecordKey& key, std::uint64_t probe, std::uint64_t present);
    Entry& Place(const DrawRecordKey& key, std::uint64_t probe, std::uint64_t present, std::uint64_t front);
    Installed Install(const std::shared_ptr<const DrawRecord>& record, std::uint64_t present, std::uint64_t front);
    bool Demote(const DrawRecordKey& key, std::uint64_t probe, std::uint64_t front);
    void Retire(Entry& entry, std::uint64_t front);
    void Clear(std::uint64_t front);
    void Release(std::uint64_t front);
    void Reset();

    std::size_t Identities() const { return identities; }
    std::size_t Records() const { return records; }
    std::size_t Bytes() const { return bytes; }
    std::size_t RetiredSets() const { return retired.size(); }
    std::size_t ByteCap() const { return byteCap; }
    void SetByteCap(std::size_t cap) { byteCap = cap; }
    static std::size_t SetOf(std::uint64_t probe);

private:
    Entry* locate(const DrawRecordKey& key, std::uint64_t probe);
    void sweep(std::uint64_t present, std::uint64_t front, const Entry* keep);
    void noteGauges() const;

    std::vector<Entry> entries;
    std::vector<std::pair<std::shared_ptr<const DrawRecordSet>, std::uint64_t>> retired;
    std::size_t identities = 0;
    std::size_t records = 0;
    std::size_t bytes = 0;
    std::size_t byteCap = DefaultByteCap;
    std::uint64_t uses = 0;
    std::uint64_t sweptPresent = ~std::uint64_t{0};
};

}

#endif
