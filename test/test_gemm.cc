#include <gtest/gtest.h>

#include "gemm/cpu_gemm.hpp"
#include "utilities/read_csv.hpp"

TEST(GEMM_TEST, CPU_native_test) {
    // Read input data
    auto A = utilities::read_csv<double>("/tmp/matrix_a_128x128.csv");
    auto B = utilities::read_csv<double>("/tmp/matrix_b_128x128.csv");
    auto C = utilities::read_csv<double>("/tmp/matrix_c_128x128.csv");

    EXPECT_EQ(A.size(), B[0].size());

    // Compute reference result
    std::vector<std::vector<double> > ref_result(C.size(), std::vector<double>(C[0].size(), 0.0));
    gemm::cpu_native_gemm(A, B, ref_result);

    EXPECT_NEAR(ref_result[1][10], C[1][10], 1e-3);
    EXPECT_NEAR(ref_result[22][44], C[22][44], 1e-3);
    EXPECT_NEAR(ref_result[11][66], C[11][66], 1e-3);
}

TEST(GEMM_TEST, CPU_openmp_test) {
    // Read input data
    auto A = utilities::read_csv<double>("/tmp/matrix_a_128x128.csv");
    auto B = utilities::read_csv<double>("/tmp/matrix_b_128x128.csv");
    auto C = utilities::read_csv<double>("/tmp/matrix_c_128x128.csv");

    EXPECT_EQ(A.size(), B[0].size());

    // Compute reference result
    std::vector<std::vector<double> > ref_result(C.size(), std::vector<double>(C[0].size(), 0.0));
    gemm::cpu_openmp_gemm(A, B, ref_result);

    EXPECT_NEAR(ref_result[1][10], C[1][10], 1e-3);
    EXPECT_NEAR(ref_result[22][44], C[22][44], 1e-3);
    EXPECT_NEAR(ref_result[11][66], C[11][66], 1e-3);
}

TEST(GEMM_TEST, CPU_std_parallel_test) {
    // Read input data
    auto A = utilities::read_csv<double>("/tmp/matrix_a_128x128.csv");
    auto B = utilities::read_csv<double>("/tmp/matrix_b_128x128.csv");
    auto C = utilities::read_csv<double>("/tmp/matrix_c_128x128.csv");

    EXPECT_EQ(A.size(), B[0].size());

    // Compute reference result
    std::vector<std::vector<double> > ref_result(C.size(), std::vector<double>(C[0].size(), 0.0));
    gemm::cpu_std_parallel_gemm(A, B, ref_result);

    EXPECT_NEAR(ref_result[1][10], C[1][10], 1e-3);
    EXPECT_NEAR(ref_result[22][44], C[22][44], 1e-3);
    EXPECT_NEAR(ref_result[11][66], C[11][66], 1e-3);
}