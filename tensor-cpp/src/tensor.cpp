#include <cassert>
#include "tensor.h"

void matmul_naive(const Tensor2D& A, const Tensor2D& B, Tensor2D& C) {
    std::size_t A_rows = A.rows;
    std::size_t A_cols = A.cols;
    std::size_t B_cols = B.cols;
    assert(A_cols == B.rows && "Incompatible matrix dimensions for multiplication.");
    assert(C.rows == A_rows && C.cols == B_cols && "Output tensor has incorrect dimensions.");

    // 遍历 A 的每一行 i 和 B 的每一列 j
    for (std::size_t i = 0; i < A_rows; ++i) {
        for (std::size_t j = 0; j < B_cols; ++j) {

            // A 的第 i 行和 B 的第 j 列的点积
            float sum = 0.0f;
            for (std::size_t k = 0; k < A_cols; ++k) {
                sum += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) = sum;
        }
    }
}
