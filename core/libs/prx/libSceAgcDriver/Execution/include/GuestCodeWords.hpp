#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_GUESTCODEWORDS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_GUESTCODEWORDS_HPP

#include <cstdint>

namespace AgcDriver::GuestCode {

inline constexpr std::uint32_t LiteralSource = 255u;
inline constexpr std::uint32_t ExecLo = 126u;
inline constexpr std::uint32_t WaitVmcnt0 = 0xbf8c3f70u;
inline constexpr std::uint32_t Endpgm = 0xbf810000u;

constexpr std::uint32_t Vgpr(std::uint32_t index) {
    return 256u + index;
}

constexpr std::uint32_t Inline(std::uint32_t value) {
    return 128u + value;
}

constexpr std::uint32_t Vop1(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0) {
    return (0x3fu << 25u) | (vdst << 17u) | (op << 9u) | src0;
}

constexpr std::uint32_t Vop2(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0, std::uint32_t vsrc1) {
    return (op << 25u) | (vdst << 17u) | (vsrc1 << 9u) | src0;
}

constexpr std::uint32_t Vopc(std::uint32_t op, std::uint32_t src0, std::uint32_t vsrc1) {
    return (0x3eu << 25u) | (op << 17u) | (vsrc1 << 9u) | src0;
}

constexpr std::uint32_t Vop3Low(std::uint32_t op, std::uint32_t vdst, std::uint32_t opSel, std::uint32_t abs, bool clamp) {
    return (0x35u << 26u) | (op << 16u) | ((clamp ? 1u : 0u) << 15u) | ((opSel & 0xfu) << 11u) | ((abs & 7u) << 8u) | vdst;
}

constexpr std::uint32_t Vop3High(std::uint32_t src0, std::uint32_t src1, std::uint32_t src2, std::uint32_t neg) {
    return ((neg & 7u) << 29u) | (src2 << 18u) | (src1 << 9u) | src0;
}

constexpr std::uint32_t Vop3pLow(std::uint32_t op, std::uint32_t vdst, std::uint32_t opSel, std::uint32_t opSelHi, std::uint32_t negHi, bool clamp) {
    return (0xccu << 24u) | (op << 16u) | ((clamp ? 1u : 0u) << 15u) | (((opSelHi >> 2u) & 1u) << 14u) | ((opSel & 7u) << 11u) | ((negHi & 7u) << 8u) | vdst;
}

constexpr std::uint32_t Vop3pHigh(std::uint32_t src0, std::uint32_t src1, std::uint32_t src2, std::uint32_t opSelHi, std::uint32_t neg) {
    return ((neg & 7u) << 29u) | ((opSelHi & 3u) << 27u) | (src2 << 18u) | (src1 << 9u) | src0;
}

constexpr std::uint32_t Sop1(std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc0) {
    return (0x17du << 23u) | (sdst << 16u) | (op << 8u) | ssrc0;
}

constexpr std::uint32_t Sop2(std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc0, std::uint32_t ssrc1) {
    return (0x2u << 30u) | (op << 23u) | (sdst << 16u) | (ssrc1 << 8u) | ssrc0;
}

constexpr std::uint32_t Sopc(std::uint32_t op, std::uint32_t ssrc0, std::uint32_t ssrc1) {
    return (0x17eu << 23u) | (op << 16u) | (ssrc1 << 8u) | ssrc0;
}

constexpr std::uint32_t MubufLow(std::uint32_t op, std::uint32_t offset, bool offen) {
    return (0x38u << 26u) | ((op & 0x7fu) << 18u) | ((offen ? 1u : 0u) << 12u) | (offset & 0xfffu);
}

constexpr std::uint32_t MubufHigh(std::uint32_t vaddr, std::uint32_t vdata, std::uint32_t srsrc, std::uint32_t soffset) {
    return (soffset << 24u) | (srsrc << 16u) | (vdata << 8u) | vaddr;
}

}

#endif
