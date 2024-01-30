#include <decoder.h>
#include <huffman.h>
#include <fft.h>

#include "bitreader.h"
#include <glog/logging.h>

#include <array>
#include <cstddef>
#include <cmath>
#include <optional>
#include <unordered_map>
#include <unordered_set>

class DecoderState;

namespace {
// markers
const uint16_t kSOI = 0xFFD8;       // start
const uint16_t kEOI = 0xFFD9;       // finish
const uint16_t kCOM = 0xFFFE;       // comments
const uint16_t kAPPFirst = 0xFFE0;  // application-specific data
const uint16_t kAPPLast = 0xFFEF;
const uint16_t kDQT = 0xFFDB;   // quantization table
const uint16_t kSOF0 = 0xFFC0;  // meta about image
const uint16_t kDHT = 0xFFC4;   // huffman
const uint16_t kSOS = 0xFFDA;   // data

const uint8_t kCellN = 8;
const uint8_t kHalfSize = 4;
const uint8_t kHuffmanDepth = 16;

const uint8_t kDCid = 0;
const uint8_t kACid = 1;

// YCbCr -> RGB conversion coefficients
const double kConvA = 0.299;
const double kConvB = 0.587;
const double kConvC = 0.114;
const double kConvD = 1.772;
const double kConvE = 1.402;

const std::unordered_map<uint16_t, std::string> kMarkerName = {
    {kSOI, "SOI"},   {kEOI, "EOI"}, {kCOM, "COM"}, {kDQT, "DQT"},
    {kSOF0, "SOF0"}, {kDHT, "DHT"}, {kSOS, "SOS"},
};
};  // namespace

uint16_t MakeMarker(const std::vector<uint8_t> &bytes) {
    return (static_cast<uint16_t>(bytes[0]) << 8) | bytes[1];
}

std::pair<uint8_t, uint8_t> SplitByte(uint8_t byte) {
    return {byte >> 4, byte & ((1 << 4) - 1)};
}

template <class T>
std::vector<T> RestoreFromZigZag(size_t n, size_t m, const std::vector<T> &arr) {
    if (arr.size() != n * m) {
        throw std::invalid_argument("Zig-Zag array size must be n * m");
    }
    std::vector<T> mat(n * m);
    uint8_t i = 0;
    uint8_t j = 0;
    for (const auto &elem : arr) {
        mat[i * m + j] = elem;
        if ((i + j) % 2 == 0) {
            if (i == 0) {
                ++j;
            } else if (j + 1 == m) {
                ++i;
            } else {
                --i;
                ++j;
            }
        } else {
            if (i + 1 == n) {
                ++j;
            } else if (j == 0) {
                ++i;
            } else {
                ++i;
                --j;
            }
        }
    }
    return mat;
}

class DecoderState {
public:
    DecoderState(std::istream &input)
        : reader_(input),
          dct_input_(kCellN * kCellN),
          dct_output_(kCellN * kCellN),
          dct_(kCellN, &dct_input_, &dct_output_) {
    }

    bool ReadSection() {  // returns false if current section is EOI
        auto marker = MakeMarker(reader_.ReadBytes(2));
        //        DLOG(INFO) << marker;
        if (marker != kSOI && !flags_.ContainsMarker(kSOI)) {
            throw std::runtime_error("non-SOI marker at the start of file");
        }
        if (marker >= kAPPFirst && marker <= kAPPLast) {
            //            flags_.AddMarker(marker, true);
            ReadAPPnSection(marker);
        } else {
            switch (marker) {
                case kSOI:
                    flags_.AddMarker(marker, true);
                    break;
                case kEOI:
                    if (!flags_.ContainsMarker(kSOS)) {
                        throw std::runtime_error("EOI section must be the last one");
                    }
                    flags_.AddMarker(marker, true);
                    return false;
                    break;
                case kCOM:
                    flags_.AddMarker(marker, true);
                    ReadCOMSection();
                    break;
                case kDQT:
                    flags_.AddMarker(marker);
                    ReadDQTSection();
                    break;
                case kSOF0:
                    flags_.AddMarker(marker, true);
                    ReadSOF0Section();
                    break;
                case kDHT:
                    flags_.AddMarker(marker);
                    ReadDHTSection();
                    break;
                case kSOS:
                    flags_.AddMarker(marker, true);
                    ReadSOSSection();
                    break;
                default:
                    throw std::runtime_error("Trying to read unknown marker");
                    break;
            }
        }
        return true;
    }

    Image GetImage() {
        return image_;
    }

private:
    struct ChannelMeta {
        uint8_t h_subs_rate;
        uint8_t v_subs_rate;
        uint8_t quant_table_id;
        uint8_t huffman_dc_id;
        uint8_t huffman_ac_id;
    };

    struct StatusFlags {
        std::unordered_set<uint16_t> markers_;

        bool AddMarker(uint16_t marker, bool check_uniqueness = false) {
            if (markers_.contains(marker)) {
                if (check_uniqueness) {
                    auto section_name =
                        (kMarkerName.contains(marker) ? kMarkerName.at(marker) : "APPn");
                    throw std::runtime_error(section_name + " section must be unique");
                }
                return false;
            }
            markers_.insert(marker);
            return true;
        }

        bool ContainsMarker(uint16_t marker) {
            return markers_.contains(marker);
        }

        bool MayBeReadyForDecoding() {
            return markers_.contains(kDQT) && markers_.contains(kDHT) && markers_.contains(kSOF0);
        }
    };

    struct ProcessingMeta {
        ProcessingMeta(ChannelMeta &meta, uint8_t id)
            : id(id),
              vs(meta.v_subs_rate),
              hs(meta.h_subs_rate),
              block(vs, std::vector<std::vector<double>>(hs)) {
        }

        double GetVal(uint8_t i, uint8_t j) {
            i /= v_comp;
            j /= h_comp;
            auto local_i = i % kCellN;
            auto local_j = j % kCellN;
            return block[i / kCellN][j / kCellN][local_i * kCellN + local_j];
        }
        uint8_t id;
        uint8_t vs;      // vertical subsampling rate
        uint8_t hs;      // horizontal subsampling rate
        uint8_t v_comp;  // vertical compression
        uint8_t h_comp;  // horizontal compression
        int prev_dc = 0;
        std::vector<std::vector<std::vector<double>>> block;
    };

    BitReader reader_;
    Image image_;
    StatusFlags flags_;

    std::vector<double> dct_input_;
    std::vector<double> dct_output_;
    DctCalculator dct_;
    std::array<std::optional<std::vector<uint16_t>>, 1 << kHalfSize> dqt_tables_;
    std::array<std::optional<ChannelMeta>, 4> meta_;
    std::array<std::array<std::optional<HuffmanTree>, 1 << kHalfSize>, 2>
        huffman_trees_;  // class, id, tree

    uint16_t ReadMarker() {
        return MakeMarker(reader_.ReadBytes(2));
    }

    std::pair<uint16_t, std::vector<uint8_t>> ReadSizeAndData() {
        auto size = ReadMarker() - 2;
        return {size, reader_.ReadBytes(size)};
    }

    void ReadAPPnSection(uint16_t marker) {
        ReadSizeAndData();
    }
    void ReadCOMSection() {
        auto comm = ReadSizeAndData().second;
        image_.SetComment({comm.begin(), comm.end()});
    }
    void ReadDQTSection() {
        auto [size, data] = ReadSizeAndData();
        for (uint16_t i = 0; i < size;) {
            auto [len_id, table_id] = SplitByte(data[i++]);
            if (len_id != 0 && len_id != 1) {
                throw std::runtime_error("Length id in DQT section must be 0 or 1");
            }
            uint16_t value_len = len_id + 1;
            if (size - i < value_len * kCellN * kCellN) {
                throw std::runtime_error("Not enough data in DQT section");
            }
            std::vector<uint16_t> arr;
            arr.reserve(kCellN * kCellN);
            for (uint8_t j = 0; j < kCellN * kCellN; ++j, i += value_len) {
                arr.push_back((value_len == 1 ? static_cast<uint8_t>(data[i])
                                              : MakeMarker({data[i], data[i + 1]})));
            }
            dqt_tables_[table_id] = std::move(arr);
        }
    }
    void ReadSOF0Section() {
        auto [size, data] = ReadSizeAndData();
        if (size < 6) {
            std::runtime_error("Not enough data in SOF0 section");
        }
        uint16_t i = 0;
        uint8_t precision = data[i++];
        if (precision != 8) {
            std::runtime_error("Precision must be 8");
        }
        auto height = MakeMarker({data[i], data[i + 1]});
        i += 2;
        auto width = MakeMarker({data[i], data[i + 1]});
        i += 2;
        uint8_t channels_cnt = data[i++];
        if (size - i != channels_cnt * 3 || !(channels_cnt == 1 || channels_cnt == 3)) {
            throw std::runtime_error("Broken SOF0 section");
        }
        image_.SetSize(width, height);
        for (uint8_t j = 0; j < channels_cnt; ++j, i += 3) {
            uint8_t id = data[i];
            if (id != j + 1) {
                throw std::runtime_error("Channel id in SOF0 must be in [1; 3]");
            }
            meta_[id] = ChannelMeta();
            auto &meta = meta_[id].value();
            auto [h, v] = SplitByte(data[i + 1]);
            if ((h != 1 && h != 2) || (v != 1 && v != 2)) {
                throw std::runtime_error("Incorrect subsampling rate in SOF0");
            }
            meta.h_subs_rate = h;
            meta.v_subs_rate = v;
            meta.quant_table_id = data[i + 2];
        }
    }
    void ReadDHTSection() {
        auto [size, data] = ReadSizeAndData();
        for (uint16_t i = 0; i < size;) {
            if (size - i < 1 + kHuffmanDepth) {
                throw std::runtime_error("Not enough data in DHT section");
            }
            auto [class_id, table_id] = SplitByte(data[i++]);
            if (class_id != 0 && class_id != 1) {
                throw std::runtime_error("Class id must be 0 or 1");
            }
            huffman_trees_[class_id][table_id] = HuffmanTree();
            std::vector<uint8_t> code_lengths = {data.begin() + i,
                                                 data.begin() + i + kHuffmanDepth};
            i += kHuffmanDepth;
            uint16_t cnt = 0;
            for (auto len : code_lengths) {
                cnt += len;
            }
            if (size - i < cnt) {
                throw std::runtime_error("Not enough data in DHT section");
            }
            std::vector<uint8_t> values = {data.begin() + i, data.begin() + i + cnt};
            i += cnt;
            huffman_trees_[class_id][table_id]->Build(code_lengths, values);
        }
    }
    uint8_t HuffmanGetValue(uint8_t class_id, uint8_t table_id) {
        int value;
        auto &cur_tree = huffman_trees_.at(class_id).at(table_id).value();
        while (!cur_tree.Move(reader_.ReadBit(), value)) {
        }
        return value;
    }
    int DecodeValue(int val, uint8_t len) {
        if (len != 0 && !(val & (1 << (len - 1)))) {
            val ^= (1 << len) - 1;
            val = -val;
        }
        return val;
    }
    std::vector<double> ReadAndDecodeTable(uint8_t channel_id, int &prev_dc) {
        auto &cur_meta = meta_.at(channel_id).value();
        uint8_t dc_len = HuffmanGetValue(kDCid, cur_meta.huffman_dc_id);
        std::fill(dct_input_.begin(), dct_input_.end(), 0);
        auto &dqt = dqt_tables_.at(cur_meta.quant_table_id).value();
        int dc_diff = DecodeValue(reader_.ReadBits(dc_len), dc_len);
        prev_dc += dc_diff;
        std::vector<double> zigzag(kCellN * kCellN, 0);
        zigzag[0] = prev_dc * dqt[0];
        for (uint16_t i = 1; i < kCellN * kCellN; ++i) {
            auto [zeros, ac_len] = SplitByte(HuffmanGetValue(kACid, cur_meta.huffman_ac_id));
            i += zeros;
            if (zeros == 0 && ac_len == 0) {
                break;
            }
            zigzag[i] = DecodeValue(reader_.ReadBits(ac_len), ac_len) * dqt[i];
        }
        auto input = RestoreFromZigZag(kCellN, kCellN, zigzag);
        for (uint8_t i = 0; i < kCellN * kCellN; ++i) {
            dct_input_[i] = input[i];
        }
        dct_.Inverse();
        std::vector<double> res(kCellN * kCellN);
        for (uint8_t i = 0; i < kCellN * kCellN; ++i) {
            res[i] = dct_output_[i] + 128;
        }
        return res;
    }

    RGB YCbCrToRGB(std::tuple<double, double, double> y_cb_cr) {
        auto [y, cb, cr] = y_cb_cr;
        RGB res;
        cb -= 128;
        cr -= 128;
        res.r = std::min(255, std::max<int>(0, y + kConvE * cr));
        res.g = std::min(255, std::max<int>(0, y - (kConvA * kConvE / kConvB) * cr -
                                                   (kConvC * kConvD / kConvB) * cb));
        res.b = std::min(255, std::max<int>(0, y + kConvD * cb));
        return res;
    }
    void ReadSOSSection() {
        if (!flags_.MayBeReadyForDecoding()) {
            throw std::runtime_error("Not enough information for decoding");
        }
        auto [size, data] = ReadSizeAndData();
        if (size < 1) {
            throw std::runtime_error("Not enough data in SOS section");
        }
        uint16_t i = 0;
        uint8_t channels_cnt = data[i++];
        if (channels_cnt * 2 + 3 != size - i) {
            throw std::runtime_error("Not enough data in SOS section");
        }
        std::vector<ProcessingMeta> channels;
        uint8_t mcu_v = 0;
        uint8_t mcu_h = 0;
        for (uint8_t j = 0; j < channels_cnt; ++j, i += 2) {
            uint8_t channel_id = data[i];
            auto [dc_id, ac_id] = SplitByte(data[i + 1]);
            auto &cur_meta = meta_.at(channel_id).value();
            cur_meta.huffman_dc_id = dc_id;
            cur_meta.huffman_ac_id = ac_id;
            channels.emplace_back(cur_meta, channel_id);
            mcu_v = std::max(mcu_v, channels.back().vs);
            mcu_h = std::max(mcu_h, channels.back().hs);
        }
        if (data[i] != 0x00 || data[i + 1] != 0x3F || data[i + 2] != 0x00) {
            throw std::runtime_error("Corrupted SOS metadata");
        }
        for (auto &channel : channels) {
            channel.v_comp = mcu_v / channel.vs;
            channel.h_comp = mcu_h / channel.hs;
        }
        uint16_t mcu_v_cnt = (image_.Height() + kCellN * mcu_v - 1) / (kCellN * mcu_v);
        uint16_t mcu_h_cnt = (image_.Width() + kCellN * mcu_h - 1) / (kCellN * mcu_h);
        size_t image_height = image_.Height();
        size_t image_width = image_.Width();
        size_t offset_i = 0;
        size_t offset_j = 0;

        for (uint16_t mcu_row = 0; mcu_row < mcu_v_cnt; ++mcu_row, offset_i += mcu_v * kCellN) {
            offset_j = 0;
            for (uint16_t mcu_col = 0; mcu_col < mcu_h_cnt; ++mcu_col, offset_j += mcu_h * kCellN) {
                for (auto &channel : channels) {
                    for (auto &row : channel.block) {
                        for (auto &cell : row) {
                            cell = ReadAndDecodeTable(channel.id, channel.prev_dc);
                        }
                    }
                }

                for (uint16_t ii = 0; ii < mcu_v * kCellN && offset_i + ii < image_height; ++ii) {
                    for (uint16_t jj = 0; jj < mcu_h * kCellN && offset_j + jj < image_width;
                         ++jj) {
                        if (channels_cnt == 1) {
                            int val = channels[0].GetVal(ii, jj);
                            image_.SetPixel(offset_i + ii, offset_j + jj, {val, val, val});
                        } else {
                            image_.SetPixel(
                                offset_i + ii, offset_j + jj,
                                YCbCrToRGB({channels[0].GetVal(ii, jj), channels[1].GetVal(ii, jj),
                                            channels[2].GetVal(ii, jj)}));
                        }
                    }
                }
            }
        }
    }
};

Image Decode(std::istream &input) {
    DecoderState state(input);
    while (state.ReadSection()) {
    }
    return state.GetImage();
}
