#pragma once
#include <cstddef>  // size_t
#include <memory>

struct Tensor2D {
    std::unique_ptr<float[]> data;
    std::size_t rows;
    std::size_t cols;

    Tensor2D(std::size_t r, std::size_t c) 
        : data(std::make_unique<float[]>(r * c)), rows(r), cols(c) {}
    
    void zero() {
        for (std::size_t i = 0; i < rows * cols; ++i) {
            data[i] = 0.0f;
        }
    }

    float& at(std::size_t r, std::size_t c) {
        return data[r * cols + c];
    }

    const float& at(std::size_t r, std::size_t c) const {
        return data[r * cols + c];
    }
};

void matmul_naive(const Tensor2D& A, const Tensor2D& B, Tensor2D& C);
