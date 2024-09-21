#pragma once

#include <algorithm>
#include <execution>
#include <iostream>
#include <numeric>
#include <vector>

namespace transpose {

template <typename T>
void cpu_naive_transpose(const std::vector<std::vector<T>>& input, std::vector<std::vector<T>>& output) {
    const size_t rows = input.size();
    const size_t cols = input[0].size();

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            output[j][i] = input[i][j];  // copy element from input to output
        }
    }
}

template <typename T>
void cpu_openmp_transpose(const std::vector<std::vector<T>>& input, std::vector<std::vector<T>>& output) {
    const size_t rows = input.size();
    const size_t cols = input[0].size();

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            output[j][i] = input[i][j];  // copy element from input to output
        }
    }
}

}  // namespace transpose