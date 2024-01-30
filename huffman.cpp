#include <huffman.h>
#include <set>

const size_t kMaxTreeHeight = 16;

class TreeObserver {
public:
    TreeObserver(size_t height) : cur_height_(height), cur_mask_(0) {
    }
    std::pair<uint16_t, size_t> Move(bool bit) {
        if (cur_height_ == 0) {
            throw std::invalid_argument("Pointer is already on the lowest possible level");
        }
        --cur_height_;
        cur_mask_ |= (bit << cur_height_);
        return {cur_mask_, cur_height_};
    }

private:
    size_t cur_height_;
    uint16_t cur_mask_;
};

class HuffmanTree::Impl {
public:
    Impl(size_t height, std::vector<uint8_t> values)
        : values_(std::move(values)), height_(height), observer_(height) {
    }
    void ResetObserver() {
        observer_ = TreeObserver(height_);
    }
    std::vector<std::pair<uint16_t, size_t>> leafs_;
    std::vector<uint8_t> values_;
    size_t height_;
    TreeObserver observer_;
};

HuffmanTree::HuffmanTree() = default;

void HuffmanTree::Build(const std::vector<uint8_t> &code_lengths,
                        const std::vector<uint8_t> &values) {
    if (code_lengths.size() > kMaxTreeHeight) {
        throw std::invalid_argument("Code lengths vector size is greater than expected");
    }
    int height = code_lengths.size();
    while (height > 0 && code_lengths[height - 1] == 0) {
        if (code_lengths[height - 1] == 0) {
            --height;
        }
    }
    impl_ = std::make_unique<HuffmanTree::Impl>(height, values);

    std::set<uint16_t> empty_vertices;
    empty_vertices.insert(0);
    for (int i = 0, cur_shift = height - 1; i < height; ++i, --cur_shift) {
        for (auto it = empty_vertices.begin(); it != empty_vertices.end(); ++it) {
            it = empty_vertices.insert(*it | (1 << cur_shift)).first;
        }
        if (empty_vertices.size() < code_lengths[i]) {
            throw std::invalid_argument("There is no Huffman tree that satisfies such an input");
        }
        for (int j = 0; j < code_lengths[i]; ++j) {
            auto it = empty_vertices.begin();
            impl_->leafs_.emplace_back(*it, cur_shift);
            empty_vertices.erase(it);
        }
    }

    if (impl_->leafs_.size() != values.size()) {
        throw std::invalid_argument(
            "The number of provided values does not match with code lengths");
    }
}

bool HuffmanTree::Move(bool bit, int &value) {
    if (!impl_) {
        throw std::invalid_argument("This tree was not built before");
    }
    auto mask = impl_->observer_.Move(bit);
    auto it = std::lower_bound(impl_->leafs_.begin(), impl_->leafs_.end(), mask);
    if (it != impl_->leafs_.end() && *it == mask) {
        value = impl_->values_[it - impl_->leafs_.begin()];
        impl_->ResetObserver();
        return true;
    }
    return false;
}

HuffmanTree::HuffmanTree(HuffmanTree &&) = default;

HuffmanTree &HuffmanTree::operator=(HuffmanTree &&) = default;

HuffmanTree::~HuffmanTree() = default;
