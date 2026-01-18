#include <cassert>
#include "tensor.h"

void matmul_naive(const Tensor2D& A, const Tensor2D& B, Tensor2D& C) {
    assert(A.cols == B.rows && "Incompatible matrix dimensions for multiplication.");
    assert(C.rows == A.rows && C.cols == B.cols && "Output tensor has incorrect dimensions.");
    for (std::size_t i = 0; i < A.rows; ++i) {
        for (std::size_t j = 0; j < B.cols; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < A.cols; ++k) {
                sum += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) = sum;
        }
    }
}