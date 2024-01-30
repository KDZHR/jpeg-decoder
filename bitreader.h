#pragma once

#include <iostream>
#include <vector>
#include <cstdint>

class BitReader {
public:
    BitReader(std::istream& input);
    bool ReadBit();
    uint16_t ReadBits(uint8_t cnt);              // cnt must be <= 16
    std::vector<uint8_t> ReadBytes(size_t cnt);  // current state must be aligned
private:
    std::istream& input_;
    uint8_t buf_;
    uint8_t rem_cnt_ = 0;  // number of bits in the buffer
};