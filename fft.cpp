#include <fft.h>

#include <cmath>
#include <fftw3.h>

class DctCalculator::Impl {
public:
    Impl(size_t width, std::vector<double> *input, std::vector<double> *output)
        : width_(width), input_(input) {
        plan_ = fftw_plan_r2r_2d(width, width, input->data(), output->data(), FFTW_REDFT01,
                                 FFTW_REDFT01, FFTW_ESTIMATE);
    }

    void Execute() {
        for (size_t i = 0; i < width_; ++i) {
            (*input_)[i] *= M_SQRT2;
        }
        for (size_t j = 0; j < width_; ++j) {
            (*input_)[j * width_] *= M_SQRT2;
        }
        for (size_t i = 0; i < width_ * width_; ++i) {
            (*input_)[i] /= 16;
        }
        fftw_execute(plan_);
    }

    ~Impl() {
        fftw_destroy_plan(plan_);
    }

private:
    fftw_plan plan_;
    size_t width_;
    std::vector<double> *input_;
};

DctCalculator::DctCalculator(size_t width, std::vector<double> *input,
                             std::vector<double> *output) {
    if (input->size() != output->size() || width * width != input->size()) {
        throw std::invalid_argument("Vector sizes and width do not match");
    }
    impl_ = std::make_unique<Impl>(width, input, output);
}

void DctCalculator::Inverse() {
    impl_->Execute();
}

DctCalculator::~DctCalculator() = default;
