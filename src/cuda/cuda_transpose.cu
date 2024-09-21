#include <cuda_runtime.h>

#include <cassert>
#include <iostream>

#include "transpose/cuda_transpose.hpp"

namespace transpose {

namespace kernel {

template <typename T>
__global__ void native_kernel(const T* A, T* B, int N, int M) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < M) {
        B[col * N + row] = A[row * M + col];
    }
}

template <typename T, int kTileSize>
__global__ void shared_kernel(const T* A, T* B, int N, int M) {
    // Shared memory for tile of A
    __shared__ T tile_A[kTileSize][kTileSize];
    // Index of current thread in the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Index of current block in the grid
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * kTileSize + ty;
    int col = bx * kTileSize + tx;

    if (row >= N && col >= M) {
        return;
    }

    // Load tile of A into shared memory
    tile_A[ty][tx] = A[row * M + col];
    __syncthreads();
    // Write tile of B into global memory

    int row_b = bx * kTileSize + ty;
    int col_b = by * kTileSize + tx;

    B[row_b * M + col_b] = tile_A[tx][ty];
}

}  // namespace kernel

template <typename T>
void cuda_transpose_cu(const T* A, T* B, int N, int M, CudaTransposeAlgorithm algorithm) {
    if (algorithm == CudaTransposeAlgorithm::kNative) {
        dim3 block(32, 16);
        dim3 grid((N + block.y - 1) / block.y, (M + block.x - 1) / block.x);
        kernel::native_kernel<<<grid, block>>>(A, B, N, M);
    } else if (algorithm == CudaTransposeAlgorithm::kShared) {
        constexpr int kTileSize = 32;
        dim3 block(kTileSize, kTileSize);
        dim3 grid((N + block.y - 1) / block.y, (M + block.x - 1) / block.x);
        kernel::shared_kernel<T, kTileSize><<<grid, block>>>(A, B, N, M);
    } else {
        assert(false);
    }

    cudaDeviceSynchronize();
}

template <typename T>
void cuda_transpose(const std::vector<std::vector<T>>& A, std::vector<std::vector<T>>& B, CudaTransposeAlgorithm algorithm) {
    int N = A.size();
    int M = A[0].size();

    size_t size_A = N * M * sizeof(T);
    size_t size_B = M * N * sizeof(T);

    T* d_A;
    T* d_B;

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);

    // Copy data to device
    for (int i = 0; i < N; ++i) {
        cudaMemcpy(d_A + i * A[i].size(), A[i].data(), A[i].size() * sizeof(T), cudaMemcpyHostToDevice);
    }

    cuda_transpose_cu(d_A, d_B, N, M, algorithm);

    // Copy data back to host
    for (int i = 0; i < N; ++i) {
        cudaMemcpy(B[i].data(), d_B + i * M, M * sizeof(T), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_A);
    cudaFree(d_B);
}

template void cuda_transpose<float>(const std::vector<std::vector<float>>& A, std::vector<std::vector<float>>& B, CudaTransposeAlgorithm algorithm);
template void cuda_transpose<double>(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B,
                                     CudaTransposeAlgorithm algorithm);

template void cuda_transpose_cu<float>(const float* A, float* B, int N, int M, CudaTransposeAlgorithm algorithm);
template void cuda_transpose_cu<double>(const double* A, double* B, int N, int M, CudaTransposeAlgorithm algorithm);

}  // namespace transpose
