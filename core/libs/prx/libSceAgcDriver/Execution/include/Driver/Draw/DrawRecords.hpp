#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWRECORDS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWRECORDS_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRecordReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
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

struct DrawRecordTarget {
    std::uint32_t position = 0;
    std::uint32_t binding = 0;
    std::uint32_t word = 0;
};

struct DrawRecordBuffer {
    std::uint32_t binding = 0;
    std::uint32_t element = 0;
    std::uint32_t position = 0;
};

struct DrawRecordForward {
    static constexpr std::uint32_t Push = ~0u;
    std::uint32_t word = 0;
    std::uint32_t slot = 0;
    std::uint32_t binding = Push;
};

struct DrawRecordStage {
    ShaderRecompiler::ShaderStage stage = ShaderRecompiler::ShaderStage::Vertex;
    std::uint32_t pushConstantOffset = 0;
    std::uint32_t hiddenUserWords = 0;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
    std::shared_ptr<const std::vector<ShaderRecompiler::RegisteredImageView>> registeredViews;
    std::shared_ptr<DispatchVariant> variant;
    std::vector<ShaderRecompiler::MemoryRegion> regions;
    std::vector<std::int8_t> runBases;
    std::uint64_t relocatableWords = 0;
    std::uint64_t valueWords = 0;
    std::uint64_t baseValueWords = 0;
    std::uint64_t walkedWords = 0;
    std::uint64_t valueUseWords = 0;
    std::uint64_t calleeWords = 0;
    std::uint64_t forwardedWords = 0;
    std::uint64_t vertexWords = 0;
    std::uint64_t userDataMask = 0;
    std::vector<std::array<std::uint32_t, 3>> valueSlots;
    std::vector<std::array<std::uint32_t, 3>> baseValueSlots;
    std::vector<DrawRecordTarget> targets;
    std::vector<DrawRecordBuffer> buffers;
    std::vector<DrawRecordForward> forwards;
    std::vector<std::uint32_t> userData;
    std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertexInfo;
};

struct DrawRecordPlan {
    std::vector<DrawRecordStage> stages;
    std::size_t bytes = 0;
    std::uint64_t signature = 0;
    bool addressed = false;
};

struct DrawRecordProbe {
    std::uint64_t address = 0;
    std::uint32_t value = 0;
    std::uint16_t stage = 0;
    std::int16_t base = -1;
};

struct DrawRecordStageView {
    std::uint64_t variant = 0;
    std::uint32_t stage = 0;
    std::uint32_t pushOffset = 0;
    std::vector<std::uint32_t> bindings;
    std::vector<std::uint32_t> vertexAttributes;
    std::vector<ShaderRecompiler::RegisteredImageView> views;
};

struct DrawRecordView {
    std::vector<DrawRecordStageView> stages;
    std::array<std::byte, Graphics::PipelinePushConstantBytes> push{};
    bool pushed = false;
    std::array<std::uint64_t, 8> parameters{};
    std::vector<DrawPipeline::Range> reads;
    std::vector<DrawPipeline::Range> writes;
    std::vector<std::uint32_t> contentKey;
    std::optional<std::uint32_t> fragmentOutputs;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> snapshotWords;
    std::vector<std::uint64_t> fetched;

    void CaptureSnapshots(std::span<const Graphics::GuestMemorySnapshot> snapshots);
    void Capture(std::span<const Graphics::CompiledShader> shaders, const Pm4::DrawParameters& drawParameters);
};

enum class DrawRecordPlanRefusal : std::uint8_t { None, Shape, Indirect, Stages, GeometryBack, Capture, Direct, Program, Result, Patched };

struct DrawRecordSink {
    bool queued = false;
    bool plan = false;
    bool verify = false;
    DrawRecordPlanRefusal refusal = DrawRecordPlanRefusal::None;
    std::uint64_t shape = 0;
    std::vector<std::uint32_t> contentKey;
    std::vector<std::uint32_t> vertexLayout;
    std::vector<std::uint32_t> vertexInputs;
    std::optional<std::uint32_t> fragmentOutputs;
    std::vector<DrawPipeline::Range> writes;
    std::shared_ptr<const DrawRecordPlan> built;
    DrawRecordView today;

    void Reset();
    void NoteVertices(std::span<const std::optional<ShaderRecompiler::ShaderVertexStageInfo>> infos);
    DrawRecordHash Content() const { return HashDrawRecordWords(contentKey, vertexLayout); }
    DrawRecordHash Inputs() const { return HashDrawRecordWords(vertexInputs, {}); }
};

void DrawVertexLayout(std::span<const std::optional<ShaderRecompiler::ShaderVertexStageInfo>> infos, std::vector<std::uint32_t>& layout, std::vector<std::uint32_t>& inputs);

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
    std::shared_ptr<const DrawRecordPlan> plan;
    std::uint64_t signature = 0;
    std::vector<DrawRecordProbe> probes;
    std::shared_ptr<DrawRecordCommit> commit = std::make_shared<DrawRecordCommit>();

    static std::shared_ptr<DrawRecord> Build(const DrawRecordKey& key, std::shared_ptr<const LeanShape> shape, std::uint64_t deviceSerial, std::uint64_t generation, std::uint64_t present, const DrawRecordSink& sink);
    std::size_t Bytes() const;
};

struct DrawRecordSet {
    std::vector<std::shared_ptr<const DrawRecord>> records;
    std::size_t bytes = 0;

    const LeanShape* Shape() const;
    const DrawRecord* Match(const DrawRecordHash& content) const;
};

std::vector<DrawRecordProbe> DrawRecordProbes(const DrawRecordPlan& plan, const DrawRecordSet* kept);
bool DrawRecordProbesHold(std::span<const DrawRecordProbe> probes, std::span<const std::vector<std::uint32_t>> users);
std::optional<std::size_t> DrawRecordTwin(const DrawRecordSet& set, const DrawRecordHash& content, std::uint64_t generation, std::uint64_t deviceSerial, const std::uint64_t* signature);

struct DrawRecordAnswer {
    std::shared_ptr<const DrawRecord> built;
    DrawRecordKey key;
    bool demote = false;
};

struct DrawRecordHit {
    std::shared_ptr<const DrawRecord> record;
    std::vector<ShaderRecompiler::HeapStageWords> sides;
    std::vector<Graphics::CompiledShader> stages;
    Pm4::DrawParameters parameters{};
    std::vector<DrawPipeline::Range> reads;
    std::vector<std::vector<std::uint32_t>> words;
    std::vector<std::vector<Graphics::DecodeRead>> decodeReads;
    std::vector<Graphics::GuestMemorySnapshot> snapshots;
    std::vector<std::uint64_t> fetched;
    bool inputsMoved = false;
    bool baseValueMoved = false;
    bool buffersPatched = false;
    bool relocated = false;
    bool planned = false;
    bool direct = false;
    // A direct hit's class, which keeps the record and the stages' donors alive, and the captures
    // whose registered image views the hit's stages point into.
    std::shared_ptr<const void> owner;
    std::vector<std::shared_ptr<const ShaderRecompiler::ResourceCapture>> captures;
    // A direct hit's buffers prepared on the finisher (APS5_DIRECT_BUFFERS); the moved words point
    // into `stages`, so they stay valid until the commit ran.
    std::optional<Graphics::ShaderResources::PreparedBuffers> preparedBuffers;
    std::optional<Graphics::PreparedInputs> preparedInputs;
    std::optional<DrawFragment> fragment;
    Graphics::ShaderResources::PreparedBuffers spareBuffers;
    Graphics::PreparedInputs spareInputs;
    bool pooled = false;
};

namespace DrawRecordFlags {

constexpr std::uint8_t On = 1;
constexpr std::uint8_t Probed = 2;
constexpr std::uint8_t BuildWanted = 4;
constexpr std::uint8_t VerifyDue = 8;
constexpr std::uint8_t StagePlans = 16;
constexpr std::uint8_t PlanProbed = 32;
constexpr std::uint8_t Direct = 64;
constexpr std::uint8_t DirectBuffers = 128;
constexpr std::uint16_t DirectInputs = 256;

}

bool ReadRecordUserWords(const DrawProgram& program, const DrawUserWords& words, std::vector<std::uint32_t>& out);
bool SameBufferFields(const std::uint32_t* before, const std::uint32_t* after);

class DrawRecordStore {
public:
    static constexpr std::size_t Ways = 4;
    static constexpr std::size_t Sets = 8192;
    static constexpr std::size_t Variants = 8;
    static constexpr std::size_t Candidates = 2;
    static constexpr std::size_t ProbesPerVariant = 2;
    static constexpr std::size_t MaxProbes = 8;
    static constexpr std::uint64_t KeepPresents = 3;
    static constexpr std::size_t DefaultByteCap = std::size_t{512} << 20;
    static std::size_t ConfiguredByteCap();

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
    std::size_t byteCap = ConfiguredByteCap();
    std::uint64_t uses = 0;
    std::uint64_t sweptPresent = ~std::uint64_t{0};
};

}

#endif
