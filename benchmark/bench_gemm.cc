#include <benchmark/benchmark.h>

#include <vector>

#include "gemm/cpu_gemm.hpp"
#include "gemm/cuda_gemm.hpp"
#include "utilities/generate_matrix.hpp"
static void BM_cpu_transpose_native_gemm(benchmark::State& state) {
    float lower_bound = 0.0;
    float upper_bound = 1.0;

    auto A = utilities::generate_random_matrix<float, 2>(lower_bound, upper_bound, {state.range(0), state.range(1)});
    auto B = utilities::generate_random_matrix<float, 2>(lower_bound, upper_bound, {state.range(0), state.range(1)});

    std::vector<std::vector<float>> C(A.size(), std::vector<float>(B.size(), 0.0f));

    for (auto _ : state) {
        gemm::cpu_native_gemm(A, B, C, true);
        benchmark::DoNotOptimize(C);
    }
}

// Native GEMM
static void BM_cpu_native_gemm(benchmark::State& state) {
    float lower_bound = 0.0;
    float upper_bound = 1.0;

    auto A = utilities::generate_random_matrix<float, 2>(lower_bound, upper_bound, {state.range(0), state.range(1)});
    auto B = utilities::generate_random_matrix<float, 2>(lower_bound, upper_bound, {state.range(1), state.range(0)});

    std::vector<std::vector<float>> C(A.size(), std::vector<float>(B[0].size(), 0.0f));

    for (auto _ : state) {
        gemm::cpu_native_gemm(A, B, C);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_cpu_transpose_openmp_gemm(benchmark::State& state) {
    float lower_bound = 0.0;
    float upper_bound = 1.0;

    auto A = utilities::generate_random_matrix<float, 2>(lower_bound, upper_bound, {state.range(0), state.range(1)});
    auto B = utilities::generate_random_matrix<float, 2>(lower_bound, upper_bound, {state.range(0), state.range(1)});

    std::vector<std::vector<float>> C(A.size(), std::vector<float>(B.size(), 0.0f));

    for (auto _ : state) {
        gemm::cpu_openmp_gemm(A, B, C, true);
        benchmark::DoNotOptimize(C);
    }
}

// OpenMP GEMM
static void BM_cpu_openmp_gemm(benchmark::State& state) {
    float lower_bound = 0.0;
    float upper_bound = 1.0;

    auto A = utilities::generate_random_matrix<float, 2>(lower_bound, upper_bound, {state.range(0), state.range(1)});
    auto B = utilities::generate_random_matrix<float, 2>(lower_bound, upper_bound, {state.range(1), state.range(0)});

    std::vector<std::vector<float>> C(A.size(), std::vector<float>(B[0].size(), 0.0f));

    for (auto _ : state) {
        gemm::cpu_openmp_gemm(A, B, C);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_cpu_transpose_parallel_for_each_gemm(benchmark::State& state) {
    float lower_bound = 0.0;
    float upper_bound = 1.0;

    auto A = utilities::generate_random_matrix<float, 2, int64_t>(lower_bound, upper_bound, {state.range(0), state.range(1)});

    auto B = utilities::generate_random_matrix<float, 2, int64_t>(lower_bound, upper_bound, {state.range(0), state.range(1)});

    std::vector<std::vector<float>> C(A.size(), std::vector<float>(B.size(), 0.0f));

    for (auto _ : state) {
        gemm::cpu_std_parallel_gemm(A, B, C, true);
        benchmark::DoNotOptimize(C);
    }
}

// Parallel std::for_each GEMM
static void BM_cpu_parallel_for_each_gemm(benchmark::State& state) {
    float lower_bound = 0.0;
    float upper_bound = 1.0;

    auto A = utilities::generate_random_matrix<float, 2, int64_t>(lower_bound, upper_bound, {state.range(0), state.range(1)});

    auto B = utilities::generate_random_matrix<float, 2, int64_t>(lower_bound, upper_bound, {state.range(1), state.range(0)});

    std::vector<std::vector<float>> C(A.size(), std::vector<float>(B[0].size(), 0.0f));

    for (auto _ : state) {
        gemm::cpu_std_parallel_gemm(A, B, C);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_gpu_native_gemm(benchmark::State& state) {
    float lower_bound = 0.0;
    float upper_bound = 1.0;

    auto A = utilities::generate_random_matrix<float, 2, int64_t>(lower_bound, upper_bound, {state.range(0), state.range(1)});

    auto B = utilities::generate_random_matrix<float, 2, int64_t>(lower_bound, upper_bound, {state.range(1), state.range(0)});

    std::vector<std::vector<float>> C(A.size(), std::vector<float>(B[0].size(), 0.0f));

    for (auto _ : state) {
        gemm::cuda_gemm(A, B, C, gemm::CudaGemmAlgorithm::kNative);
        benchmark::DoNotOptimize(C);
    }
}

BENCHMARK(BM_cpu_transpose_native_gemm)->Args({1024, 2048})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_cpu_native_gemm)->Args({1024, 2048})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_cpu_transpose_openmp_gemm)->Args({1024, 2048})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_cpu_openmp_gemm)->Args({1024, 2048})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_cpu_transpose_parallel_for_each_gemm)->Args({1024, 2048})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_cpu_parallel_for_each_gemm)->Args({1024, 2048})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_gpu_native_gemm)->Args({1024, 2048})->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
