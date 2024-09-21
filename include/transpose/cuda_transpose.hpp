#pragma once

#include <vector>

namespace transpose {

enum class CudaTransposeAlgorithm { kNative, kShared };

template <typename T>
void cuda_transpose(const std::vector<std::vector<T>>& A, std::vector<std::vector<T>>& B, CudaTransposeAlgorithm algorithm);

template <typename T>
void cuda_transpose_cu(const T* A, T* B, int N, int M, CudaTransposeAlgorithm algorithm);

}  // namespace transpose