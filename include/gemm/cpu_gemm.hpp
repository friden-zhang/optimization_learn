#pragma once

#include <algorithm>
#include <execution>
#include <iostream>
#include <numeric>
#include <vector>
namespace gemm {

template <typename T>
void cpu_native_gemm(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B, std::vector<std::vector<T>>& C,
                     bool transpose_b = false) {
    if (!transpose_b) {
        for (int i = 0; i < A.size(); i++) {
            for (int j = 0; j < B[0].size(); j++) {
                for (int k = 0; k < B.size(); k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    } else {
        for (int i = 0; i < A.size(); i++) {
            for (int j = 0; j < B.size(); j++) {
                for (int k = 0; k < B[0].size(); k++) {
                    C[i][j] += A[i][k] * B[j][k];
                }
            }
        }
    }
}

template <typename T>
void cpu_openmp_gemm(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B, std::vector<std::vector<T>>& C,
                     bool transpose_b = false) {
    // Using collapse(2) combines the outer two loops (i and j) into a single parallel loop.
    // This approach allows each thread to handle the computation of a row or block of rows in the
    // output matrix C. It strikes a balance between maximizing parallelism and minimizing overhead
    // from thread management and synchronization. Using collapse(3) would combine all three loops
    // (i, j, and k), resulting in each thread computing a single element C[i][j]. Although
    // collapse(3) can potentially increase parallelism, it often leads to significant overhead from
    // thread management and synchronization, especially when the computation per element is
    // relatively small.
    if (!transpose_b) {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < A.size(); i++) {
            for (int j = 0; j < B[0].size(); j++) {
                T local_element = 0;
                for (int k = 0; k < B.size(); k++) {
                    local_element += A[i][k] * B[k][j];
                }
                C[i][j] = local_element;
            }
        }
    } else {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < A.size(); i++) {
            for (int j = 0; j < B.size(); j++) {
                T local_element = 0;
                for (int k = 0; k < B[0].size(); k++) {
                    local_element += A[i][k] * B[j][k];
                }
                C[i][j] = local_element;
            }
        }
    }
}

template <typename T>
void cpu_std_parallel_gemm(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B, std::vector<std::vector<T>>& C,
                           bool transpose_b = false) {
    if (!transpose_b) {
        std::for_each(std::execution::par, C.begin(), C.end(), [&](std::vector<T>& row) {
            size_t i = &row - &C[0];
            std::for_each(std::execution::par, row.begin(), row.end(), [&](T& element) {
                size_t j = &element - &row[0];
                T local_element = 0;
                for (int k = 0; k < B.size(); ++k) {
                    local_element += A[i][k] * B[k][j];
                }
                element = local_element;
            });
        });
    } else {
        std::for_each(std::execution::par, C.begin(), C.end(), [&](std::vector<T>& row) {
            size_t i = &row - &C[0];
            std::for_each(std::execution::par, row.begin(), row.end(), [&](T& element) {
                size_t j = &element - &row[0];
                T local_element = 0;
                for (int k = 0; k < B[0].size(); ++k) {
                    local_element += A[i][k] * B[j][k];
                }
                element = local_element;
            });
        });
    }
}

}  // namespace gemm