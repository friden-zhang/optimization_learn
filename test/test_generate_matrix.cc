#include <gtest/gtest.h>

#include "utilities/generate_matrix.hpp"

TEST(GenerateMatrix, Test) {
    auto matrix = utilities::generate_random_matrix<float, 3>(0.0f, 1.0f, {128, 512, 1024});
    EXPECT_EQ(matrix.size(), 128);
    EXPECT_EQ(matrix[0].size(), 512);
    EXPECT_EQ(matrix[0][0].size(), 1024);
}