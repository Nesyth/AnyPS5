#include "prx/libSceAgcDriver/Graphics/include/DevicePageTable.hpp"
#include <cstdio>
#include <string>
#include <vector>

using AgcDriver::Graphics::DevicePageTable;
namespace Abi = ShaderRecompiler::BdaAbi;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

const char* Name(DevicePageTable::Cover cover) {
    switch (cover) {
    case DevicePageTable::Cover::Covered: return "covered";
    case DevicePageTable::Cover::Missing: return "missing";
    case DevicePageTable::Cover::OtherAddress: return "other address";
    case DevicePageTable::Cover::Narrower: return "narrower";
    }
    return "unknown";
}

void ExpectCover(std::span<const Abi::Range> table, std::span<const Abi::Range> build, DevicePageTable::Cover expected, const std::string& what) {
    const auto cover = DevicePageTable::Compare(table, build);
    Expect(cover == expected, what + ": " + Name(cover) + " instead of " + Name(expected));
}

void CompareTests() {
    using Cover = DevicePageTable::Cover;
    constexpr auto rw = Abi::Read | Abi::Write;
    const std::vector<Abi::Range> table{
        {0x10000, 0x20000, 0x900000, rw, 0},
        {0x20000, 0x30000, 0x910000, Abi::Read, 0},
        {0x40000, 0x50000, 0xa00000, rw, 0},
    };
    ExpectCover(table, std::vector<Abi::Range>{}, Cover::Covered, "an empty build");
    ExpectCover(table, std::vector<Abi::Range>{{0x11000, 0x12000, 0x901000, Abi::Read, 0}}, Cover::Covered, "a range inside one entry");
    ExpectCover(table, std::vector<Abi::Range>{{0x18000, 0x28000, 0x908000, Abi::Read, 0}}, Cover::Covered, "a range across two contiguous entries");
    ExpectCover(table, std::vector<Abi::Range>{{0x28000, 0x41000, 0x918000, Abi::Read, 0}}, Cover::Missing, "a range over the gap");
    ExpectCover(table, std::vector<Abi::Range>{{0x60000, 0x61000, 0xb00000, Abi::Read, 0}}, Cover::Missing, "a range past the table");
    ExpectCover(table, std::vector<Abi::Range>{{0x8000, 0x11000, 0x8f8000, Abi::Read, 0}}, Cover::Missing, "a range starting before the table");
    ExpectCover(table, std::vector<Abi::Range>{{0x11000, 0x12000, 0x123000, Abi::Read, 0}}, Cover::OtherAddress, "a range at another device address");
    ExpectCover(table, std::vector<Abi::Range>{{0x21000, 0x22000, 0x911000, rw, 0}}, Cover::Narrower, "a write into a read-only entry");
    ExpectCover(table, std::vector<Abi::Range>{{0x41000, 0x42000, 0xa01000, rw, 0}, {0x11000, 0x11000, 0x1, rw, 0}}, Cover::Covered, "an empty range in the build");
    ExpectCover(std::vector<Abi::Range>{}, std::vector<Abi::Range>{{0x11000, 0x12000, 0x901000, Abi::Read, 0}}, Cover::Missing, "an empty table");
}

}

int main() {
    CompareTests();
    if (failures != 0) {
        std::fprintf(stderr, "device page table tests: %d failures\n", failures);
        return 1;
    }
    std::printf("device page table tests passed\n");
    return 0;
}
