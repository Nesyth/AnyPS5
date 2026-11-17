#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYCENSUS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYCENSUS_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {
class Buffer;
}

namespace AgcDriver::Graphics::ResidencyCensus {

extern const bool Enabled;
inline bool On() { return Enabled; }
std::uint64_t LineSampleInterval();

enum class Miss : std::uint8_t { Hit = 0, First, Rebuilt, Lost, Unarmed, Gap, Layout, Uncovered, Import, Unwatched, IndexBreak, Bumped, CpuAtMarks, Shadow, Pending, CpuWriter, QueuedStore, Count };

inline const char* MissName(Miss miss) {
    constexpr std::array<const char*, static_cast<std::size_t>(Miss::Count)> names{"hit", "first", "rebuilt", "lost", "unarmed", "gap", "layout", "uncovered", "import", "unwatched", "index-break", "bumped", "cpu-at-marks", "shadow", "pending", "cpu-writer", "queued-store"};
    const auto index = static_cast<std::size_t>(miss);
    return index < names.size() ? names[index] : "unnamed";
}

struct UseFacts {
    bool seen = false;
    bool sameBuild = false;
    bool copiedBack = false;
    bool armed = false;
    bool gap = false;
    bool layout = false;
    bool covered = false;
    bool sameImport = false;
    bool watched = false;
    bool indexBreak = false;
    bool bumped = false;
    bool cpuAtMarks = false;
    bool shadow = false;
    bool pending = false;
    bool cpuWriter = false;
    bool queuedStore = false;
};

inline Miss FirstMiss(const UseFacts& facts, bool shared) {
    if (!facts.seen) return Miss::First;
    if (!shared && !facts.sameBuild) return Miss::Rebuilt;
    if (!facts.copiedBack) return Miss::Lost;
    if (!facts.armed) return Miss::Unarmed;
    if (facts.gap) return Miss::Gap;
    if (!facts.layout) return Miss::Layout;
    if (!facts.covered) return Miss::Uncovered;
    if (!facts.sameImport) return Miss::Import;
    if (!facts.watched) return Miss::Unwatched;
    if (facts.indexBreak) return Miss::IndexBreak;
    if (facts.bumped) return Miss::Bumped;
    if (facts.cpuAtMarks) return Miss::CpuAtMarks;
    if (facts.shadow) return Miss::Shadow;
    if (facts.pending) return Miss::Pending;
    if (facts.cpuWriter) return Miss::CpuWriter;
    if (facts.queuedStore) return Miss::QueuedStore;
    return Miss::Hit;
}

constexpr std::uint64_t LineBytes = 128;
constexpr std::uint64_t LineWords = LineBytes / 4u;
constexpr std::uint32_t GroupWords = 128;

inline std::uint64_t LinesSpanned(std::uint64_t address, std::uint64_t bytes) {
    if (bytes == 0) return 0;
    return (address + bytes - 1u) / LineBytes - address / LineBytes + 1u;
}

inline std::uint32_t LeadWords(std::uint64_t address) {
    return static_cast<std::uint32_t>((address % LineBytes) / 4u);
}

struct LineCount {
    std::uint64_t changedLines = 0;
    std::uint64_t changedWords = 0;
};

constexpr std::uint32_t UnwrittenWord = ~std::uint32_t{0};

inline LineCount CountChangedLines(std::uint32_t lead, std::span<const std::uint32_t> result, std::span<const std::uint32_t> snapshot) {
    LineCount count;
    bool any = false;
    std::uint64_t line = 0;
    const auto words = std::min(result.size(), snapshot.size());
    for (std::size_t index = 0; index < words; ++index) {
        if (result[index] == snapshot[index]) continue;
        ++count.changedWords;
        const auto at = (std::uint64_t{lead} + index) / LineWords;
        if (any && at == line) continue;
        any = true;
        line = at;
        ++count.changedLines;
    }
    return count;
}

struct LineDispatch {
    std::uint32_t base = 0;
    std::uint32_t groups = 0;
};

inline std::vector<LineDispatch> LineDispatches(std::uint32_t lead, std::uint32_t words, std::uint32_t maxGroups) {
    std::vector<LineDispatch> dispatches;
    if (words == 0 || maxGroups == 0) return dispatches;
    const std::uint64_t total = std::uint64_t{lead} + words;
    const std::uint64_t step = std::uint64_t{maxGroups} * GroupWords;
    for (std::uint64_t base = 0; base < total; base += step) {
        const auto chunk = std::min(total - base, step);
        dispatches.push_back({static_cast<std::uint32_t>(base), static_cast<std::uint32_t>((chunk + GroupWords - 1u) / GroupWords)});
    }
    return dispatches;
}

enum class Stamper : std::uint8_t { None = 0, Dispatch = 1, Cmask = 2, Copy = 3, Fill = 4, SampleDump = 5, ImageStore = 6, Publish = 7, Htile = 8, StagedMark = 16, DirectMark = 17, Restamp = 18, ImageWriteBack = 19 };
constexpr std::uint64_t TagWhoMask = (std::uint64_t{1} << 56u) - 1u;
constexpr std::uint64_t Tag(Stamper kind, std::uint64_t who) {
    return (std::uint64_t{static_cast<std::uint8_t>(kind)} << 56u) | (who & TagWhoMask);
}

inline const char* StamperName(std::uint64_t tag) {
    switch (static_cast<Stamper>(static_cast<std::uint8_t>(tag >> 56u))) {
    case Stamper::None:
        return "untagged";
    case Stamper::Dispatch:
        return "dispatch";
    case Stamper::Cmask:
        return "cmask";
    case Stamper::Copy:
        return "copy";
    case Stamper::Fill:
        return "fill";
    case Stamper::SampleDump:
        return "sample-dump";
    case Stamper::ImageStore:
        return "image-store";
    case Stamper::Publish:
        return "shadow-publish";
    case Stamper::Htile:
        return "htile";
    case Stamper::StagedMark:
        return "staged-mark";
    case Stamper::DirectMark:
        return "direct-mark";
    case Stamper::Restamp:
        return "restamp";
    case Stamper::ImageWriteBack:
        return "image-write-back";
    }
    return "other";
}

class StampScope {
public:
    StampScope(Stamper kind, std::uint64_t who) noexcept {
        if (!On()) return;
        previous = enter(kind, who);
        active = true;
    }
    ~StampScope() {
        if (active) leave(previous);
    }
    StampScope(const StampScope&) = delete;
    StampScope& operator=(const StampScope&) = delete;

private:
    static std::uint64_t enter(Stamper kind, std::uint64_t who) noexcept;
    static void leave(std::uint64_t tag) noexcept;
    std::uint64_t previous = 0;
    bool active = false;
};

std::uint64_t NewBuild();

struct Use {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::uint64_t build = 0;
    bool reused = false;
    bool copiedBack = false;
    std::uint64_t sourceAddress = 0;
    std::uint64_t sourceBase = 0;
    bool watched = false;
    bool shadow = false;
    bool pending = false;
    bool cpuWriter = false;
    bool queuedStore = false;
    std::uint64_t shadowSerial = 0;
    std::uint64_t recorder = 0;
    std::uint64_t completedThrough = 0;
};
std::uint32_t NoteUse(const Use& use);

struct Arm {
    std::uint32_t region = 0;
    std::uint64_t build = 0;
    bool layout = false;
    bool covered = false;
    bool copyBackOwed = false;
    std::uint64_t shadowSerial = 0;
};
void NoteArm(const Arm& arm);

bool SampleLines(std::uint32_t region);
void NoteLineSample(std::uint32_t region, std::uint64_t recorder, std::uint64_t serial, std::uint64_t lines, std::shared_ptr<Buffer> readback);

void NoteWriteBack(std::uint64_t begin, std::uint64_t end, const char* reason, bool blockUnits, std::uint64_t slabBytes, std::uint64_t importBytes, std::uint64_t cpuBytes, std::uint64_t keptBlocks);
std::uint64_t ThreadSlabBytes();
void NoteFlush(const char* reason, std::size_t images, std::uint64_t slabBytes, std::uint64_t publishedUnits);
void NoteUpload(std::uint64_t base, std::span<const std::pair<std::uint64_t, std::uint64_t>> runs, const char* reason);
enum class DropCause : std::uint8_t { ClearKeys = 0, Compared, Stamped, Count };
void NoteDrop(std::uint64_t begin, std::uint64_t bytes, std::uint64_t generation, std::uint64_t pendingSince, DropCause cause);
void NoteHookRead(std::uint64_t address, std::size_t bytes, std::uint8_t site);
void NoteAddressBuild(std::uint64_t begin, std::uint64_t end);

enum class CopyOrigin : std::uint8_t { Unknown = 0, Capture, CpuUpload, Unwatch, StoredSave, CpuWriteBack, Restore, Count };

struct CopyClass {
    std::uint64_t images = 0;
    std::uint64_t bytes = 0;
    void Add(std::uint64_t more) {
        ++images;
        bytes += more;
    }
};

struct Copies {
    bool release = false;
    std::uint64_t cachedImages = 0;
    CopyClass holding;
    CopyClass holdsAll;
    CopyClass holdsSome;
    CopyClass holdsNone;
    std::uint64_t released = 0;
    CopyClass keptToVerify;
    CopyClass watched;
    CopyClass unwatched;
    CopyClass imported;
    CopyClass notImported;
    CopyClass pending;
    CopyClass idle;
    CopyClass baselines;
    CopyClass saved;
    std::array<CopyClass, static_cast<std::size_t>(CopyOrigin::Count)> origins{};
};
bool CopiesDue();
void ReportCopies(const Copies& copies);

}

#endif
