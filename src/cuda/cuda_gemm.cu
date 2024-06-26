#include <cuda_runtime.h>

#include <cassert>

#include "gemm/cuda_gemm.hpp"

namespace gemm {

namespace kernel {

template <typename T>
__global__ void native_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= K) {
        return;
    }

    T value = 0.0f;
    for (int e = 0; e < N; ++e) {
        value += A[row * N + e] * B[e * K + col];
    }
    C[row * K + col] = value;
}

}  // namespace kernel

template <typename T>
void cuda_gemm(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B, std::vector<std::vector<T>>& C,
               CudaGemmAlgorithm algorithm) {
    int M = C.size();
    int K = C[0].size();
    int N = B.size();

    size_t sizeA = M * N * sizeof(T);
    size_t sizeB = N * K * sizeof(T);
    size_t sizeC = M * K * sizeof(T);

    T *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Copy data to device
    for (int i = 0; i < M; ++i) {
        cudaMemcpy(d_A + i * A[i].size(), A[i].data(), A[i].size() * sizeof(T), cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < N; ++i) {
        cudaMemcpy(d_B + i * B[i].size(), B[i].data(), B[i].size() * sizeof(T), cudaMemcpyHostToDevice);
    }

    // Launch kernel
    dim3 block(32, 16);
    dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    if (algorithm == CudaGemmAlgorithm::kNative) {
        kernel::native_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    } else {
        // TODO: Add other algorithms here
        assert(false);
    }
    cudaDeviceSynchronize();
    // copy result back to host
    for (int i = 0; i < M; ++i) {
        cudaMemcpy(C[i].data(), d_C + i * K, K * sizeof(T), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

template void cuda_gemm<float>(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B, std::vector<std::vector<float>>& C,
                               CudaGemmAlgorithm algorithm);
template void cuda_gemm<double>(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B,
                                std::vector<std::vector<double>>& C, CudaGemmAlgorithm algorithm);

}  // namespace gemm