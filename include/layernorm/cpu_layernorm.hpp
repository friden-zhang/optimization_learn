#pragma once

#include <algorithm>
#include <numeric>

namespace layernorm {

template <typename T, typename U = T>
void cpu_layernorm(const T* input, T* output, const T* alpha, const T* beta, int N, int M) {
    const U epsilon = 1e-5;

#pragma omp parallel for num_threads(4)
    for (int row = 0; row < N; row++) {
        const T* row_input = static_cast<const U*>(input + row * M);
        T* row_output = static_cast<U*>(output + row * M);

        const U mean = std::accumulate(row_input, row_input + M, U(0)) / M;
        U var = std::accumulate(row_input, row_input + M, U(0), [mean](U acc, U x) { return acc + x * x; }) / M;
        var -= mean * mean;
        const U inv_std = 1 / std::sqrt(var + epsilon);

        if (alpha == nullptr && beta == nullptr) {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < M; i++) {
                row_output[i] = static_cast<T>((row_input[i] - mean) * inv_std);
            }
        } else if (alpha == nullptr && beta != nullptr) {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < M; i++) {
                row_output[i] = static_cast<T>((row_input[i] - mean) * inv_std + beta[row]);
            }
        } else if (beta == nullptr && alpha != nullptr) {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < M; i++) {
                row_output[i] = static_cast<T>(alpha[row]) * (row_input[i] - mean) * inv_std;
            }
        } else {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < M; i++) {
                row_output[i] = static_cast<T>(alpha[row] * (row_input[i] - mean) * inv_std + beta[row]);
            }
        }
    }
}

}  // namespace layernorm