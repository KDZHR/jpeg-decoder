#include "bitreader.h"
#include <glog/logging.h>

namespace {
const char kMarker = 0xFF;
const char kEscapeChar = 0x00;
}  // namespace

BitReader::BitReader(std::istream &input) : input_(input) {
}

bool BitReader::ReadBit() {
    if (rem_cnt_ == 0) {
        char cur;
        input_.read(&cur, 1);
        if (cur == kMarker) {
            char next_sym;
            input_.read(&next_sym, 1);
            if (next_sym != kEscapeChar) {
                throw std::runtime_error("Trying to read bits of a marker");
            }
        }
        buf_ = cur;
        rem_cnt_ = 8;
    }
    return (buf_ >> (--rem_cnt_)) & 1;
}

std::vector<uint8_t> BitReader::ReadBytes(size_t cnt) {
    std::vector<uint8_t> bytes(cnt);
    std::string buf;
    buf.resize(cnt);
    input_.read(buf.data(), cnt);
    for (size_t i = 0; i < cnt; ++i) {
        bytes[i] = buf[i];
    }
    return bytes;
}

uint16_t BitReader::ReadBits(uint8_t cnt) {
    uint16_t res = 0;
    for (uint8_t i = 0; i < cnt; ++i) {
        res = (res << 1) | ReadBit();
    }
    return res;
}
