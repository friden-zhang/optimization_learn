#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>

#include "layernorm/cuda_layernorm.hpp"

namespace layernorm {

template <typename U>
struct U2 {
    U x;
    U y;
    __device__ U2() {}
    __device__ U2(U x, U y) : x(x), y(y) {}
};

template <typename U>
static __device__ __forceinline__ U2<U> warp_reduce_sum(U2<U> a) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        a.x += __shfl_xor_sync(0xffffffff, a.x, mask, 32);
        a.y += __shfl_xor_sync(0xffffffff, a.y, mask, 32);
    }
    return a;
}

template <int block_size, typename T, typename U>
__global__ void layernorm_kernel(const T* input, T* output, const T* alpha, const T* beta, int N, int M) {
    // NOTICE: block size caclulate [1, M], rowwise block to [tid, 1]
    // NOTICE: grid size max be (N1, N2), N = N1 * N2
    const int row = blockIdx.y * gridDim.x + blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row >= N) {
        return;
    }

    U2<U> mean_var{U(0), U(0)};

    for (int i = tid; i < M; i += block_size) {
        const int index = row * M + i;
        const U x = static_cast<U>(input[index]);
        mean_var.x += x;
        mean_var.y += x * x;
    }

    mean_var = warp_reduce_sum(mean_var);

    // use shared memory to store mean and variance, because block_size > warp_size
    if (block_size > 32) {
        // max threads is 32x32, which is 1024
        __shared__ U2<U> shared_mean_var[32];
        int warp_id = tid / 32;
        int lane_id = tid % 32;

        // only the first thread in each warp loads the data
        if (lane_id == 0) {
            shared_mean_var[warp_id] = mean_var;
        }

        __syncthreads();
        mean_var = shared_mean_var[lane_id];
        mean_var = warp_reduce_sum(mean_var);
    }

    // now mn and var contain the mean and variance of the block
    const U mn = mean_var.x / M;
    const U var = mean_var.y / M - mn * mn;
    const U inv_std = rsqrt(var + 1e-5);

    if (alpha == nullptr && beta == nullptr) {
        for (int i = tid; i < M; i += block_size) {
            const int index = row * M + i;
            output[index] = static_cast<T>(inv_std * (static_cast<U>(input[index]) - mn));
        }
    } else if (alpha == nullptr && beta != nullptr) {
        for (int i = tid; i < M; i += block_size) {
            const int index = row * M + i;
            output[index] = static_cast<T>(inv_std * (static_cast<U>(input[index]) - mn) + static_cast<U>(beta[row]));
        }
    } else if (alpha != nullptr && beta == nullptr) {
        for (int i = tid; i < M; i += block_size) {
            const int index = row * M + i;
            output[index] = static_cast<T>(static_cast<U>(alpha[row]) * inv_std * (static_cast<U>(input[index]) - mn));
        }
    } else {
        for (int i = tid; i < M; i += block_size) {
            const int index = row * M + i;
            output[index] = static_cast<T>(static_cast<U>(alpha[row]) * inv_std * (static_cast<U>(input[index]) - mn) + static_cast<U>(beta[row]));
        }
    }
}

template <typename T, typename U>
bool lauch_layernorm_kernel(const T* input, T* output, const T* alpha, const T* beta, int N, int M) {
    if (M % 32 != 0) {
        std::cerr << "M must be divisible by 32 in lauch_layernorm_kernel" << std::endl;
        return false;
    }
    if (M < 1024) {
        const dim3 block_size(32, 1, 1);
        const dim3 grid_size(N, 1, 1);
        layernorm_kernel<32, T, U><<<grid_size, block_size>>>(input, output, alpha, beta, N, M);
    } else {
        const dim3 block_size(1024, 1, 1);
        const dim3 grid_size(N, 1, 1);
        layernorm_kernel<1024, T, U><<<grid_size, block_size>>>(input, output, alpha, beta, N, M);
    }
    return true;
}

}  // namespace layernorm

template bool layernorm::lauch_layernorm_kernel<float, float>(const float* input, float* output, const float* alpha, const float* beta, int N, int M);
template bool layernorm::lauch_layernorm_kernel<uint8_t, float>(const uint8_t* input, uint8_t* output, const uint8_t* alpha, const uint8_t* beta,
                                                                int N, int M);
