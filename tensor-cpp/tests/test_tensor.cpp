#include <gtest/gtest.h>
#include "tensor.h"

// 测试 Tensor 初始化
TEST(TensorTest, Initialization) {
    Tensor2D tensor(3, 4);
    EXPECT_EQ(tensor.rows, 3);
    EXPECT_EQ(tensor.cols, 4);
    tensor.zero();
    for (std::size_t i = 0; i < tensor.rows; ++i) {
        for (std::size_t j = 0; j < tensor.cols; ++j) {
            EXPECT_FLOAT_EQ(tensor.at(i, j), 0.0f);
        }
    }
}

// 测试 Tensor 元素访问和修改
TEST(TensorTest, ElementAccess) {
    Tensor2D tensor(2, 2);
    tensor.at(0, 0) = 1.0f;
    tensor.at(0, 1) = 2.0f;
    tensor.at(1, 0) = 3.0f;
    tensor.at(1, 1) = 4.0f;

    EXPECT_FLOAT_EQ(tensor.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(tensor.at(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(tensor.at(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(tensor.at(1, 1), 4.0f);
}

// 测试矩阵乘法
TEST(TensorTest, MatMulNaive) {
    Tensor2D A(2, 2);
    A.at(0, 0) = 1.0f; A.at(0, 1) = 2.0f;
    A.at(1, 0) = 3.0f; A.at(1, 1) = 4.0f;

    Tensor2D B(2, 2);
    B.at(0, 0) = 5.0f; B.at(0, 1) = 6.0f;
    B.at(1, 0) = 7.0f; B.at(1, 1) = 8.0f;

    Tensor2D C(2, 2);
    matmul_naive(A, B, C);

    EXPECT_FLOAT_EQ(C.at(0, 0), 19.0f);
    EXPECT_FLOAT_EQ(C.at(0, 1), 22.0f);
    EXPECT_FLOAT_EQ(C.at(1, 0), 43.0f);
    EXPECT_FLOAT_EQ(C.at(1, 1), 50.0f);
}
