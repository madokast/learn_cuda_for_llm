#include <cassert>
#include <cstring> // memset
#include <string_view>
#include <iostream>
#include "tensor.h"


void matmul_naive(const Tensor2D& A, const Tensor2D& B, Tensor2D& C) noexcept {
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

void matmul_naive_ijk(const Tensor2D& A, const Tensor2D& B, Tensor2D& C, std::string_view ijk) noexcept {
    std::size_t A_rows = A.rows;
    std::size_t A_cols = A.cols;
    std::size_t B_cols = B.cols;
    assert(A_cols == B.rows && "Incompatible matrix dimensions for multiplication.");
    assert(C.rows == A_rows && C.cols == B_cols && "Output tensor has incorrect dimensions.");

    std::memset(C.data.get(), 0, A_rows * C.cols * sizeof(float));

    if (ijk == "ijk") {
        for (std::size_t i = 0; i < A_rows; ++i) {
            for (std::size_t j = 0; j < B_cols; ++j) {
                float c_ij = 0;
                for (std::size_t k = 0; k < A_cols; ++k) {
                    c_ij += A.at(i, k) * B.at(k, j);
                }
                C.at(i, j) = c_ij;
            }
        }
    } else if (ijk == "ikj") {
         for (std::size_t i = 0; i < A_rows; ++i) {
           for (std::size_t k = 0; k < A_cols; ++k) {
            float a_ij = A.at(i, k);
            for (std::size_t j = 0; j < B_cols; ++j) {
                    C.at(i, j) += a_ij * B.at(k, j);
                }
            }
        }
    } else if (ijk == "jik") {
        for (std::size_t j = 0; j < B_cols; ++j) {
        for (std::size_t i = 0; i < A_rows; ++i) {
                for (std::size_t k = 0; k < A_cols; ++k) {
                    C.at(i, j) += A.at(i, k) * B.at(k, j);
                }
            }
        }
    } else if (ijk == "jki") {
        for (std::size_t j = 0; j < B_cols; ++j) {
            for (std::size_t k = 0; k < A_cols; ++k) {
        for (std::size_t i = 0; i < A_rows; ++i) {
                    C.at(i, j) += A.at(i, k) * B.at(k, j);
                }
        }
        }
    } else if (ijk == "kij") {
        for (std::size_t k = 0; k < A_cols; ++k) {
            for (std::size_t i = 0; i < A_rows; ++i) {
                for (std::size_t j = 0; j < B_cols; ++j) {
                    C.at(i, j) += A.at(i, k) * B.at(k, j);
                }
            }
        }
    } else if (ijk == "kji") {
        for (std::size_t k = 0; k < A_cols; ++k) {
            for (std::size_t j = 0; j < B_cols; ++j) {
            for (std::size_t i = 0; i < A_rows; ++i) {
                    C.at(i, j) += A.at(i, k) * B.at(k, j);
                }
            }
        }
    } else {
        std::cerr << "Invalid ijk: " << ijk << std::endl;
        std::abort(); // 立即终止，不进行清理
    }    
}
