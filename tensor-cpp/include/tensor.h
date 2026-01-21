#pragma once
#include <cstddef>  // size_t
#include <memory>
#include <cstring>
#include <string_view>
#include "utils/random.h"

struct Tensor2D {
    std::unique_ptr<float[]> data;
    std::size_t rows;
    std::size_t cols;

    Tensor2D(std::size_t r, std::size_t c) noexcept
        : data(std::make_unique<float[]>(r * c)), rows(r), cols(c) {}

    // 填充随机数
    void random(uint64_t seed) noexcept {
        FastRandom rng(seed);
        rng.fill_float_array(data.get(), rows * cols);
    }
    
    // 填充零值
    void zero() {
        std::memset(data.get(), 0, rows * cols * sizeof(float));
    }

    float& at(std::size_t r, std::size_t c) noexcept {
        return data[r * cols + c];
    }

    const float& at(std::size_t r, std::size_t c) const noexcept {
        return data[r * cols + c];
    }
};

void matmul_naive(const Tensor2D& A, const Tensor2D& B, Tensor2D& C) noexcept;
void matmul_naive_ijk(const Tensor2D& A, const Tensor2D& B, Tensor2D& C, std::string_view ijk) noexcept;
