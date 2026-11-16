#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_COMMANDSTREAM_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_COMMANDSTREAM_HPP

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

struct CommandStreamWriter;

class CommandStream {
    struct Record;

public:
    static constexpr std::size_t ChunkBytes = 64u * 1024u;
    static constexpr std::size_t LargeBytes = ChunkBytes / 4u;
    static constexpr std::size_t RetainedBytes = 1024u * 1024u;
    static constexpr std::size_t SlabBytes = 64u * 1024u;

    struct Slab {
        explicit Slab(std::size_t bytes) : data(new std::byte[bytes]), capacity(bytes) {}
        std::unique_ptr<std::byte[]> data;
        std::size_t capacity;
    };

    class Fragment {
    public:
        Fragment() = default;
        Fragment(Fragment&& other) noexcept : first(std::exchange(other.first, nullptr)), tail(std::exchange(other.tail, nullptr)), ops(std::exchange(other.ops, 0u)), bytes(std::exchange(other.bytes, 0u)), slabs(std::move(other.slabs)) {}
        Fragment& operator=(Fragment&& other) noexcept {
            if (this == &other) return *this;
            first = std::exchange(other.first, nullptr);
            tail = std::exchange(other.tail, nullptr);
            ops = std::exchange(other.ops, 0u);
            bytes = std::exchange(other.bytes, 0u);
            slabs = std::move(other.slabs);
            other.slabs.clear();
            return *this;
        }
        Fragment(const Fragment&) = delete;
        Fragment& operator=(const Fragment&) = delete;
        bool Empty() const { return first == nullptr; }
        std::uint64_t Ops() const { return ops; }
        std::uint64_t Bytes() const { return bytes; }
        std::size_t Slabs() const { return slabs.size(); }

    private:
        friend class CommandStream;
        Record* first = nullptr;
        Record** tail = nullptr;
        std::uint64_t ops = 0;
        std::uint64_t bytes = 0;
        std::vector<std::shared_ptr<Slab>> slabs;
    };

    enum class Mode : std::uint8_t { Chunks, Fragments };

    CommandStream();
    explicit CommandStream(Mode mode);
    ~CommandStream();
    CommandStream(const CommandStream&) = delete;
    CommandStream& operator=(const CommandStream&) = delete;

    static bool IsHandle(VkCommandBuffer commands) {
        return (reinterpret_cast<std::uintptr_t>(commands) & HandleCheckMask) == HandleTag;
    }

    static CommandStream* FromHandle(VkCommandBuffer commands) {
        return IsHandle(commands) ? reinterpret_cast<CommandStream*>(reinterpret_cast<std::uintptr_t>(commands) & HandleAddressMask) : nullptr;
    }

    static bool Records(const char* name);

    VkCommandBuffer Handle() const { return handle; }
    void Replay(VkCommandBuffer commands) const;
    void Close() { closed = true; }
    bool Closed() const { return closed; }
    void Reset();
    bool Empty() const { return first == nullptr; }
    std::uint64_t Ops() const { return ops; }
    std::uint64_t Bytes() const { return used; }
    std::size_t ReservedBytes() const { return reserved; }
    std::size_t SplicedSlabs() const { return spliced.size(); }

    void BeginFragment();
    Fragment EndFragment();
    void Splice(Fragment&& fragment);
    static void ReplayFragment(const Fragment& fragment, VkCommandBuffer commands);
    static bool SameRecords(const Fragment& fragment, const CommandStream& stream, std::uint64_t fromOp, bool ignoreHandles, const char** firstDifference);
    std::uint64_t MarkOps() const { return ops; }

private:
    friend struct CommandStreamWriter;
    struct Chunk;
    struct Record {
        Record* next;
        std::uint8_t op;
        std::uint8_t slot;
    };

    static constexpr std::uintptr_t HandleTag = (std::uintptr_t{0xA5C3u} << 48u) | 1u;
    static constexpr std::uintptr_t HandleCheckMask = (std::uintptr_t{0xFFFFu} << 48u) | 1u;
    static constexpr std::uintptr_t HandleAddressMask = ((std::uintptr_t{1} << 48u) - 1u) & ~std::uintptr_t{1};

    void* allocate(std::size_t bytes) {
        bytes = (bytes + 7u) & ~std::size_t{7};
        used += bytes;
        if (static_cast<std::size_t>(limit - cursor) < bytes) return grow(bytes);
        auto* const result = cursor;
        cursor += bytes;
        return result;
    }

    void link(Record* record) {
        *tail = record;
        tail = &record->next;
        ++ops;
    }

    void* grow(std::size_t bytes);
    void* growSlab(std::size_t bytes);
    Chunk* makeChunk(std::size_t capacity, Chunk* next);
    void freeChunks(Chunk* chunk);

    VkCommandBuffer handle = VK_NULL_HANDLE;
    std::byte* cursor = nullptr;
    std::byte* limit = nullptr;
    Chunk* chunks = nullptr;
    Chunk** unused = &chunks;
    Record* first = nullptr;
    Record** tail = &first;
    std::uint64_t ops = 0;
    std::uint64_t used = 0;
    std::size_t reserved = 0;
    bool closed = false;
    Mode mode = Mode::Chunks;
    std::shared_ptr<Slab> slab;
    std::vector<std::shared_ptr<Slab>> touched;
    std::vector<std::shared_ptr<Slab>> spliced;
};

void NotePipelineDynamicStates(VkPipeline pipeline, std::span<const VkDynamicState> states);
void ForgetPipelineDynamicStates(VkPipeline pipeline) noexcept;

}

#endif
