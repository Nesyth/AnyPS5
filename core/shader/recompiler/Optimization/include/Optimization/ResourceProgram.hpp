#ifndef CORE_SHADER_RECOMPILER_OPTIMIZATION_RESOURCEPROGRAM_HPP
#define CORE_SHADER_RECOMPILER_OPTIMIZATION_RESOURCEPROGRAM_HPP

#include "Recompiler.hpp"
#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/SrtWalker.hpp"
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace ShaderRecompiler {

struct PreparedControlFlow;
struct SourceEntry;
class ShaderPreparationContext {
public:
    [[nodiscard]] std::shared_ptr<const PreparedControlFlow> AcquireFrontend(const RecompileRequest& request);
    [[nodiscard]] std::uint32_t ResourcePrograms() const;

private:
    friend IrProgram PrepareResourceProgram(const RecompileRequest& request, ShaderPreparationContext* preparation);
    friend struct PreparedInterface;
    std::shared_ptr<const PreparedControlFlow> frontend;
    std::shared_ptr<SourceEntry> interfaceSource;
    std::unique_ptr<IrProgram> interfaceProgram;
    std::uint32_t resourcePrograms = 0;
};

[[nodiscard]] IrProgram PrepareResourceProgram(const RecompileRequest& request);
[[nodiscard]] IrProgram PrepareResourceProgram(const RecompileRequest& request, ShaderPreparationContext* preparation);
[[nodiscard]] std::uint64_t FrontEndBuilds();
[[nodiscard]] std::shared_ptr<const IrResourcePlan> GetResourcePlan(const RecompileRequest& request);

struct ResourceCapture {
    std::shared_ptr<const IrResourcePlan> plan;
    ResourceSnapshot snapshot;
    // The cache entry the plan belongs to; null when the request bypasses the cache.
    std::shared_ptr<SourceEntry> source;
    // The walk's read addresses when the plan has pure flat slots (the leaf of each pure slot,
    // and every other read sorted and deduplicated); empty otherwise.
    SrtReadTrace readTrace;
    SrtUserDataUse userDataUse;
    // APS5_PROFILE_DRAW: what CaptureResources spent resolving the source (the stage inputs, the
    // key over the code, the plan) before the walk.
    std::uint64_t sourceNanoseconds = 0;
};
[[nodiscard]] std::shared_ptr<const ResourceCapture> CaptureResources(const RecompileRequest& request, const SrtRuntime& runtime);

// The resolved source of a request (its cache entry with the plan built), for a driver that
// memoizes it per registered shader: ResolveSource is what CaptureResources does before the walk
// (the stage input validation, the key over the code, the lookup, the first-sight plan build), and
// the overload below captures over a handle without repeating it. A handle stays valid for every
// request with the same code and the same cache key fields (RecompileCacheKey::ContextHash plus
// the target); the vertex stages' input validation is repeated per capture because it reads V#
// fields the key does not cover. Null for a request that bypasses the cache (useCache false).
struct CompiledVariant;
struct SourceHandle {
    std::shared_ptr<SourceEntry> source;
    std::shared_ptr<const CompiledVariant> artifact;
    std::vector<std::uint64_t> staticKey;
};
// A stage result shared by every capture with the same SharedKey (the donor) and the recipe
// (MemoPatch) that turns a capture's per-draw words into the side FillSharedSide writes.
struct MemoPatch;
struct SharedStage {
    std::shared_ptr<const RecompileResult> donor;
    std::shared_ptr<const MemoPatch> patch;
    explicit operator bool() const { return donor != nullptr; }
};
struct SideForward {
    std::uint32_t binding = 0;
    std::uint32_t position = 0;
    std::uint32_t word = 0;
    bool push = false;
};
enum class SideReadKind : std::uint8_t { BufferAddress = 2, BufferSize = 3, FlatWord = 4 };
struct SideReadForward {
    std::uint32_t node = 0;
    std::uint32_t binding = 0;
    std::uint32_t position = 0;
    std::uint32_t keyMask = 0;
    SideReadKind kind = SideReadKind::BufferAddress;
};
struct SideReadBase {
    std::uint32_t parent = 0;
    std::uint32_t child = 0;
};
enum class SideNodeUse : std::uint8_t { Forward = 1, Base = 2, ReadArgument = 4, Argument = 8, Condition = 16, Clean = 32, DwordKept = 64, Uncovered = 128 };
enum class SideTape : std::uint8_t { Traced, None, UniformFill, GcnGuest };
struct SideForwards {
    bool traced = false;
    std::uint64_t addressWords = 0;
    std::vector<SideForward> forwards;
    std::vector<SideReadForward> reads;
    std::vector<SideReadForward> chainedReads;
    std::vector<SideReadBase> bases;
    std::vector<SideReadBase> children;
    std::vector<std::uint8_t> nodeUses;
    SideTape tape = SideTape::Traced;
};
class PreparedShaderInvocation {
public:
    static std::optional<PreparedShaderInvocation> TryCreate(const RecompileRequest& request, const std::shared_ptr<const SourceHandle>& handle);
    static std::optional<PreparedShaderInvocation> TryCreate(const RecompileRequest& request, const std::shared_ptr<const SourceHandle>& handle, std::span<const std::uint64_t> key);
    static std::optional<PreparedShaderInvocation> TryCreate(RecompileRequest&& request, const std::shared_ptr<const SourceHandle>& handle, std::span<const std::uint64_t> key);
    const RecompileRequest& Request() const { return request; }
    std::shared_ptr<const ResourceCapture> Capture(const SrtRuntime& runtime) const;
    std::shared_ptr<const RecompileResult> Materialize(const ResourceCapture& capture) const;
    std::shared_ptr<const RecompileResult> MaterializeHeap(const ResourceCapture& capture, HeapStageWords& words) const;
    const std::shared_ptr<const IrResourcePlan>& Plan() const;
    std::shared_ptr<const IrResourcePlan> LoadPlan() const;
    std::uint64_t VariantId() const;
    const std::shared_ptr<const SourceHandle>& Handle() const { return handle; }
    // The masked snapshot hash with the vertex buffers' per-draw fields left out: captures with
    // the same key share one donor. Nullopt for a capture of another prepared shader.
    std::optional<std::uint64_t> SharedKey(const ResourceCapture& capture) const;
    // The donor for the capture's SharedKey: the masked result memo's entry when it holds one,
    // else a new result, inserted there. Empty when the result cannot be patched or the capture
    // belongs to another prepared shader. Throws what materializing the result throws.
    SharedStage ShareStage(const ResourceCapture& capture) const;
    // The capture's words that differ from the donor's, as applying the patch to a copy of the
    // donor would write them, plus the live draw's vertex attributes. False (the side then holds
    // no usable words) where that patch would throw or the donor cannot carry the capture.
    bool FillSharedSide(const ResourceCapture& capture, const RecompileResult& donor, const MemoPatch& patch, HeapStageWords& side) const;
    SideForwards TraceSideForwards(const ResourceCapture& capture, const RecompileResult& donor, const MemoPatch& patch) const;
    [[nodiscard]] std::shared_ptr<const RecompileResult> MaterializeIfMemoized(const ResourceCapture& capture, bool* contended = nullptr) const;
    [[nodiscard]] PreparedShaderInvocation Rebound(const RecompileRequest& request) const;
    [[nodiscard]] std::uint64_t SnapshotKey(const ResourceCapture& capture) const;

private:
    PreparedShaderInvocation(RecompileRequest request, const std::shared_ptr<const SourceHandle>& handle);
    RecompileRequest request;
    std::shared_ptr<const SourceHandle> handle;
};
class ShaderPending final : public std::runtime_error {
public:
    ShaderPending() : std::runtime_error("ShaderRecompiler: the shader's module is being specialized on a background thread") {}
};
class PendingShaderScope {
public:
    PendingShaderScope();
    ~PendingShaderScope();
    PendingShaderScope(const PendingShaderScope&) = delete;
    PendingShaderScope& operator=(const PendingShaderScope&) = delete;

private:
    bool previous = false;
};
struct PendingShaderCounts {
    std::uint64_t started = 0;
    std::uint64_t finished = 0;
    std::uint64_t refused = 0;
    double milliseconds = 0.0;
};
[[nodiscard]] PendingShaderCounts PendingShaderStatistics();
[[nodiscard]] std::span<const std::uint32_t> GetPreparedCode(const SourceHandle& handle);
[[nodiscard]] bool HeapFragmentEligible(const IrResourcePlan& plan);
void CompleteHeapDataWords(const RecompileResult& shared, HeapStageWords& words);
[[nodiscard]] std::shared_ptr<const SourceHandle> PrepareShader(const RecompileRequest& request);
[[nodiscard]] std::shared_ptr<const SourceHandle> PrepareShader(const RecompileRequest& request, ShaderPreparationContext* preparation);
[[nodiscard]] bool MatchesPreparedShader(const RecompileRequest& request, const SourceHandle& handle);
void BuildPreparedShaderKey(const RecompileRequest& request, std::vector<std::uint64_t>& key);
[[nodiscard]] bool MatchesPreparedShader(const RecompileRequest& request, const SourceHandle& handle, std::span<const std::uint64_t> key);
[[nodiscard]] const CompiledShaderArtifact& GetPreparedArtifact(const SourceHandle& handle);
[[nodiscard]] std::shared_ptr<const RecompileResult> RegistrationResult(const RecompileRequest& request, const SourceHandle& handle);
[[nodiscard]] std::shared_ptr<const RecompileResult> MaterializeShader(const RecompileRequest& request, const ResourceCapture& capture, const SourceHandle& handle);
[[nodiscard]] std::shared_ptr<const SourceHandle> ResolveSource(const RecompileRequest& request);
void ReleaseUndrawnResourcePlan(const SourceHandle& handle);
[[nodiscard]] std::shared_ptr<const ResourceCapture> CaptureResources(const RecompileRequest& request, const SrtRuntime& runtime, const SourceHandle& handle);

}

#endif
