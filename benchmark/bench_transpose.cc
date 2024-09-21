#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <vector>

#include "transpose/cpu_transpose.hpp"
#include "transpose/cuda_transpose.hpp"
#include "utilities/generate_matrix.hpp"

static void BM_cpu_native_transpose(benchmark::State& state) {
    float lower_bound = 0.0;
    float upper_bound = 1.0;

    auto A = utilities::generate_random_matrix<float, 2>(lower_bound, upper_bound, {state.range(0), state.range(1)});

    std::vector<std::vector<float>> B(state.range(1), std::vector<float>(state.range(0), 0.0f));

    for (auto _ : state) {
        transpose::cpu_naive_transpose(A, B);
        benchmark::DoNotOptimize(B);
    }
}

static void BM_cpu_openmp_transpose(benchmark::State& state) {
    float lower_bound = 0.0;
    float upper_bound = 1.0;

    auto A = utilities::generate_random_matrix<float, 2>(lower_bound, upper_bound, {state.range(0), state.range(1)});

    std::vector<std::vector<float>> B(state.range(1), std::vector<float>(state.range(0), 0.0f));

    for (auto _ : state) {
        transpose::cpu_openmp_transpose(A, B);
        benchmark::DoNotOptimize(B);
    }
}

static void BM_gpu_native_gemm(benchmark::State& state) {
    float lower_bound = 0.0;
    float upper_bound = 1.0;

    int N = state.range(0);
    int M = state.range(1);

    auto A = utilities::generate_random_matrix<float, 2, int64_t>(lower_bound, upper_bound, {state.range(0), state.range(1)});

    std::vector<std::vector<float>> B(state.range(1), std::vector<float>(state.range(0), 0.0f));

    size_t sizeA = state.range(0) * state.range(1) * sizeof(float);
    size_t sizeB = state.range(0) * state.range(1) * sizeof(float);

    float *d_A, *d_B;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);

    for (int i = 0; i < N; ++i) {
        cudaMemcpy(d_A + i * A[i].size(), A[i].data(), A[i].size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    for (auto _ : state) {
        transpose::cuda_transpose_cu(d_A, d_B, N, M, transpose::CudaTransposeAlgorithm::kNative);
        benchmark::DoNotOptimize(d_B);
    }
    cudaFree(d_A);
    cudaFree(d_B);
}

static void BM_gpu_shared_gemm(benchmark::State& state) {
    float lower_bound = 0.0;
    float upper_bound = 1.0;

    int N = state.range(0);
    int M = state.range(1);

    auto A = utilities::generate_random_matrix<float, 2, int64_t>(lower_bound, upper_bound, {state.range(0), state.range(1)});

    std::vector<std::vector<float>> B(state.range(1), std::vector<float>(state.range(0), 0.0f));

    size_t sizeA = state.range(0) * state.range(1) * sizeof(float);
    size_t sizeB = state.range(0) * state.range(1) * sizeof(float);

    float *d_A, *d_B;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);

    for (int i = 0; i < N; ++i) {
        cudaMemcpy(d_A + i * A[i].size(), A[i].data(), A[i].size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    for (auto _ : state) {
        transpose::cuda_transpose_cu(d_A, d_B, N, M, transpose::CudaTransposeAlgorithm::kShared);
        benchmark::DoNotOptimize(d_B);
    }
    cudaFree(d_A);
    cudaFree(d_B);
}

BENCHMARK(BM_cpu_native_transpose)->Args({4096, 4096})->Unit(benchmark::kMillisecond)->Iterations(10);
BENCHMARK(BM_cpu_openmp_transpose)->Args({4096, 4096})->Unit(benchmark::kMillisecond)->Iterations(10);
BENCHMARK(BM_gpu_native_gemm)->Args({40960, 4096})->Unit(benchmark::kMillisecond)->Iterations(10);
BENCHMARK(BM_gpu_shared_gemm)->Args({40960, 4096})->Unit(benchmark::kMillisecond)->Iterations(10);
