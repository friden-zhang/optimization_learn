#include <cuda_runtime.h>

#include <cassert>
#include <iostream>

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

    T value = T(0);
    for (int e = 0; e < N; ++e) {
        value += A[row * N + e] * B[e * K + col];
    }
    C[row * K + col] = value;
}

template <typename T, int kTileSize>
__global__ void shared_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    __shared__ T ds_A[kTileSize][kTileSize];
    __shared__ T ds_B[kTileSize][kTileSize];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= K) {
        return;  // Out-of-bounds threads exit early
    }

    T value = T(0);

    for (int e = 0; e < N; e += kTileSize) {
        // for a, a_shared_load_row is row
        // for b, b_shared_load_col is col
        int a_shared_load_col = threadIdx.x + e;
        int b_shared_load_row = threadIdx.y + e;

        // Load ds_A from global memory A
        if (a_shared_load_col < N) {
            ds_A[threadIdx.y][threadIdx.x] = A[row * N + a_shared_load_col];
        } else {
            ds_A[threadIdx.y][threadIdx.x] = T(0);  // Padding with zero
        }

        // Load ds_B from global memory B
        if (b_shared_load_row < N) {
            ds_B[threadIdx.y][threadIdx.x] = B[b_shared_load_row * K + col];
        } else {
            ds_B[threadIdx.y][threadIdx.x] = T(0);  // Padding with zero
        }

        __syncthreads();  // Ensure all threads have loaded their data

        // Compute partial result for the tile
        for (int i = 0; i < kTileSize; ++i) {
            value += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
        }

        __syncthreads();  // Ensure all threads have finished using shared memory
    }

    // Store the result back to global memory C
    C[row * K + col] = value;
}

// TODO
template <typename T, int kBlockM, int kBlockK, int kBlockN, int kThreadSizeY, int kThreadSizeX>
// A -> mxk, B -> kxn, C -> mxn
__global__ void shared_rigster_kernel(T* A, T* B, T* C, int m, int k, int n) {
    __shared__ T s_a[kBlockM][kBlockK];
    __shared__ T s_b[kBlockK][kBlockN];

    T r_c[kThreadSizeY][kThreadSizeX] = {0};

    const int kBlockYthread = kBlockM / kThreadSizeY;
    const int kBlockXthread = kBlockN / kThreadSizeX;

    const int kBlockThreadNum = kBlockYthread * kBlockXthread;

    // tid is the index of current thread in block
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    // current thread need treansform first data's index
    const int a_tile_row = tid / kBlockK;
    const int a_tile_col = tid % kBlockK;
    const int b_tile_row = tid / kBlockN;
    const int b_tile_col = tid % kBlockN;

    // many data need sride in row and col
    const int a_tile_row_stride = kBlockThreadNum / kBlockK;
    const int b_tile_row_stride = kBlockThreadNum / kBlockN;

    for (int block_k_start_index = 0; block_k_start_index < k; block_k_start_index += kBlockK) {
#pragma unroll
        for (int i = 0; i < kBlockM; i += a_tile_row_stride) {
            const int row = kBlockM * blockIdx.y + i + a_tile_row;
            const int col = block_k_start_index + a_tile_col;
            s_a[i + a_tile_row][a_tile_col] = row < m && col < k ? A[row * n + col] : 0;
        }
#pragma unroll
        for (int i = 0; i < kBlockK; i += b_tile_row_stride) {
            const int row = block_k_start_index + b_tile_row + i;
            const int col = blockIdx.x * kBlockN + b_tile_col;
            s_b[b_tile_row + i][b_tile_col] = row < k && col < n ? B[row * n + col] : 0;
        }

        __syncthreads();

        // calculate
#pragma unroll
        for (int kk = 0; kk < kBlockK; kk++) {
#pragma unroll
            for (int ty = 0; ty < kThreadSizeY; ty++) {
                for (int tx = 0; tx < kThreadSizeX; tx++) {
                    // Why
                    r_c[ty][tx] += s_a[kThreadSizeY * threadIdx.y + ty][kk] * s_b[kk][kThreadSizeX * threadIdx.x + tx];
                }
            }
        }

        __syncthreads();
    }

    // store to c
#pragma unroll
    for (int ty = 0; ty < kThreadSizeY; ty++) {
#pragma unroll
        for (int tx = 0; tx < kThreadSizeX; tx++) {
            const int row = kBlockM * blockIdx.y + kThreadSizeY * threadIdx.y + ty;
            const int col = kBlockN * blockIdx.x + kThreadSizeX * threadIdx.x + tx;
            if (row < m && col < n) {
                C[row * n + col] += r_c[ty][tx];
            }
        }
    }
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
    if (algorithm == CudaGemmAlgorithm::kNative) {
        dim3 block(32, 16);
        dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        kernel::native_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    } else if (algorithm == CudaGemmAlgorithm::kShared) {
        constexpr int kTileSize = 16;
        dim3 block(kTileSize, kTileSize);
        dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        kernel::shared_kernel<T, kTileSize><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    } else if (algorithm == CudaGemmAlgorithm::kSharedRigster) {
        constexpr int kBlockM = 128;
        constexpr int kBlockK = 8;
        constexpr int kBlockN = 128;
        // every thread compute kThreadSizeY x kThreadN matrix
        constexpr int kThreadSizeY = 8;
        constexpr int kThreadSizeX = 8;
        dim3 block(kBlockM / kThreadSizeY, kBlockN / kThreadSizeX);
        dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        kernel::shared_rigster_kernel<T, kBlockM, kBlockK, kBlockN, kThreadSizeY, kThreadSizeX><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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